#!/usr/bin/env python3
"""
SpiderBot PPO Training — pure NumPy, no PyTorch required.

Usage:
    python train_rl.py                        # train headless
    python train_rl.py --render               # train with GUI
    python train_rl.py --eval checkpoints/latest.pkl
    python train_rl.py --steps 3000000 --lr 1e-4
"""

import argparse
import os
import pickle
import time
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH  = os.path.join(SCRIPT_DIR, "spiderbot_assy", "spiderbot_assy.urdf")

# ══════════════════════════════════════════════════════════════════════════════
# Neural-network primitives (forward + backprop, no autograd)
# ══════════════════════════════════════════════════════════════════════════════

class Layer:
    """Dense layer with optional tanh activation."""

    def __init__(self, in_dim: int, out_dim: int, activation=None, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation
        self._x = self._z = self._a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        self._a = np.tanh(self._z) if self.activation == "tanh" else self._z
        return self._a

    def backward(self, grad_a: np.ndarray):
        """Returns (grad_W, grad_b, grad_x)."""
        grad_z = grad_a * (1.0 - self._a ** 2) if self.activation == "tanh" else grad_a
        return self._x.T @ grad_z, grad_z.sum(axis=0), grad_z @ self.W.T


class MLP:
    def __init__(self, sizes: list, seed: int = 0):
        self.layers = [
            Layer(sizes[i], sizes[i + 1],
                  "tanh" if i < len(sizes) - 2 else None,
                  seed + i)
            for i in range(len(sizes) - 1)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> list:
        """Returns [(grad_W, grad_b), ...] in forward layer order."""
        grads, g = [], grad_out
        for layer in reversed(self.layers):
            gW, gb, g = layer.backward(g)
            grads.insert(0, (gW, gb))
        return grads

    def get_flat(self) -> np.ndarray:
        return np.concatenate([v.ravel() for l in self.layers for v in (l.W, l.b)])

    def set_flat(self, flat: np.ndarray):
        idx = 0
        for l in self.layers:
            for arr in (l.W, l.b):
                n = arr.size
                arr.flat[:] = flat[idx: idx + n]
                idx += n

    @property
    def n_params(self) -> int:
        return sum(l.W.size + l.b.size for l in self.layers)


# ══════════════════════════════════════════════════════════════════════════════
# Adam optimizer
# ══════════════════════════════════════════════════════════════════════════════

class Adam:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self._m = self._v = None
        self._t = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)
        self._t += 1
        self._m = self.b1 * self._m + (1 - self.b1) * grads
        self._v = self.b2 * self._v + (1 - self.b2) * grads ** 2
        m_hat = self._m / (1 - self.b1 ** self._t)
        v_hat = self._v / (1 - self.b2 ** self._t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state(self):
        return {"m": self._m, "v": self._v, "t": self._t}

    def load_state(self, s):
        self._m, self._v, self._t = s["m"], s["v"], s["t"]


# ══════════════════════════════════════════════════════════════════════════════
# Actor-Critic with PPO update
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic:
    """
    Gaussian actor  μ = actor_net(obs),  σ = exp(log_std)  [shared across states]
    Value critic    V = critic_net(obs)
    All parameters updated via manual backprop + Adam.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), seed=0):
        self.act_dim = act_dim
        sizes        = [obs_dim] + list(hidden)
        self.actor   = MLP(sizes + [act_dim],  seed=seed)
        self.critic  = MLP(sizes + [1],        seed=seed + 999)
        self.log_std = np.full(act_dim, -0.5)

    # ── inference ──────────────────────────────────────────────────────────

    def value(self, obs: np.ndarray) -> np.ndarray:
        return self.critic.forward(obs).ravel()

    def act(self, obs: np.ndarray, deterministic=False):
        mean = self.actor.forward(obs)
        if deterministic:
            return mean, None
        std    = np.exp(self.log_std)
        action = mean + std * np.random.randn(*mean.shape)
        logp   = self._logp(action, mean)
        return action, logp

    def _logp(self, a, mean) -> np.ndarray:
        std = np.exp(self.log_std)
        return -0.5 * np.sum(
            ((a - mean) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi),
            axis=-1,
        )

    # ── PPO minibatch update ────────────────────────────────────────────────

    def ppo_update(self, obs, actions, old_logp, advantages, returns,
                   actor_opt: Adam, critic_opt: Adam,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad=0.5):
        B   = len(obs)
        std = np.exp(self.log_std)

        # Forward
        mean   = self.actor.forward(obs)          # (B, act_dim)
        values = self.critic.forward(obs).ravel()  # (B,)
        logp   = self._logp(actions, mean)         # (B,)

        ratio = np.exp(
            np.clip(logp - old_logp, -20.0, 20.0)
        )

        # ── policy gradient w.r.t. logp ──
        is_clipped = (
            ((ratio > 1 + clip_eps) & (advantages > 0)) |
            ((ratio < 1 - clip_eps) & (advantages < 0))
        )
        dL_dlogp = -ratio * advantages / B
        dL_dlogp[is_clipped] = 0.0

        # ── actor mean gradient: dL/dmean[i,k] = dL/dlogp[i] * (a[i,k]-mean[i,k])/std[k]^2
        # Note: d(logp)/d(mean) = (actions - mean)/std^2
        dL_dmean = dL_dlogp[:, None] * (actions - mean) / (std ** 2)  # (B, act_dim)

        # ── log_std gradient ──
        d_logp_d_logstd = (actions - mean) ** 2 / (std ** 2) - 1.0   # (B, act_dim)
        dL_dlogstd = (dL_dlogp[:, None] * d_logp_d_logstd).sum(axis=0)
        dL_dlogstd -= ent_coef   # entropy bonus: maximise H = sum(log_std) + const

        # ── value gradient ──
        dL_dv = (2.0 * vf_coef * (values - returns) / B)[:, None]    # (B, 1)

        # ── backprop ──
        actor_grads  = self.actor.backward(dL_dmean)
        critic_grads = self.critic.backward(dL_dv)

        # Flatten param grads, append log_std grads, apply Adam
        af = np.concatenate([g.ravel() for pair in actor_grads for g in pair])
        af = np.concatenate([af, dL_dlogstd])
        af = _clip_grad(af, max_grad)

        actor_full = np.concatenate([self.actor.get_flat(), self.log_std])
        actor_full = actor_opt.step(actor_full, af)
        self.actor.set_flat(actor_full[: self.actor.n_params])
        self.log_std = actor_full[self.actor.n_params:]

        cf = np.concatenate([g.ravel() for pair in critic_grads for g in pair])
        cf = _clip_grad(cf, max_grad)
        self.critic.set_flat(critic_opt.step(self.critic.get_flat(), cf))

        # Logging scalars
        clip_r      = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = float(-np.mean(np.minimum(ratio * advantages, clip_r * advantages)))
        value_loss  = float(np.mean((values - returns) ** 2))
        entropy     = float(0.5 * np.sum(1 + 2 * self.log_std + np.log(2 * np.pi)))
        return policy_loss, value_loss, entropy

    # ── serialisation ───────────────────────────────────────────────────────

    def save(self, path: str, extra=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "actor":   self.actor.get_flat(),
                "critic":  self.critic.get_flat(),
                "log_std": self.log_std.copy(),
                "extra":   extra,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.actor.set_flat(d["actor"])
        self.critic.set_flat(d["critic"])
        self.log_std = d["log_std"]
        return d.get("extra")


def _clip_grad(g: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(g)
    return g * (max_norm / n) if n > max_norm else g


# ══════════════════════════════════════════════════════════════════════════════
# SpiderBot Gymnasium environment
# ══════════════════════════════════════════════════════════════════════════════

class SpiderBotEnv(gym.Env):
    """
    Observation  (28-dim): base_pos(3) + euler(3) + linvel(3) + angvel(3)
                           + joint_pos(8) + joint_vel(8)
    Action       (8-dim):  normalised joint targets in [-1, 1]
    Reward:  forward velocity − tilt penalty − energy penalty + alive bonus
    """

    OBS_DIM  = 28
    ACT_DIM  = 8
    MAX_STEPS   = 1000
    SPAWN_Z     = 0.25
    MIN_HEIGHT  = 0.07
    MAX_HEIGHT  = 0.55
    MAX_TILT    = np.deg2rad(60)

    def __init__(self, render: bool = False):
        super().__init__()
        self._render = render
        self._client: int | None = None

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.ACT_DIM,), dtype=np.float32)

        self._joint_idx: list  = []
        self._jlimits: np.ndarray = np.empty((0, 2))
        self._robot: int = -1
        self._step_n: int = 0
        self._prev_x: float = 0.0

        self._connect()

    # ── PyBullet lifecycle ─────────────────────────────────────────────────

    def _connect(self):
        mode = p.GUI if self._render else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        if self._render:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.15],
                physicsClientId=self._client,
            )

    def _rebuild(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        self._robot = p.loadURDF(
            URDF_PATH, [0, 0, self.SPAWN_Z],
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._client,
        )
        # collect revolute joint indices and limits
        self._joint_idx, limits = [], []
        for i in range(p.getNumJoints(self._robot, physicsClientId=self._client)):
            info = p.getJointInfo(self._robot, i, physicsClientId=self._client)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_idx.append(i)
                limits.append((info[8], info[9]))
        self._jlimits = np.array(limits)

    # ── Gym API ────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rebuild()
        for _ in range(60):                         # let robot settle
            p.stepSimulation(physicsClientId=self._client)
        self._step_n  = 0
        self._prev_x  = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)[0][0]
        return self._obs().astype(np.float32), {}

    def step(self, action: np.ndarray):
        self._step_n += 1
        self._apply(action)
        p.stepSimulation(physicsClientId=self._client)
        if self._render:
            time.sleep(1.0 / 240.0)
        obs    = self._obs()
        reward = self._reward(action, obs)
        done   = self._done(obs)
        return obs.astype(np.float32), reward, done, False, {}

    def close(self):
        if self._client is not None:
            try:
                p.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = None

    # ── internals ─────────────────────────────────────────────────────────

    def _apply(self, action: np.ndarray):
        lo, hi  = self._jlimits[:, 0], self._jlimits[:, 1]
        targets = lo + (np.clip(action, -1, 1) + 1.0) * 0.5 * (hi - lo)
        for j, ji in enumerate(self._joint_idx):
            p.setJointMotorControl2(
                self._robot, ji, p.POSITION_CONTROL,
                targetPosition=float(targets[j]),
                force=50.0,
                physicsClientId=self._client,
            )

    def _obs(self) -> np.ndarray:
        c = self._client
        pos, orn  = p.getBasePositionAndOrientation(self._robot, physicsClientId=c)
        lv, av    = p.getBaseVelocity(self._robot, physicsClientId=c)
        euler     = p.getEulerFromQuaternion(orn)
        js        = [p.getJointState(self._robot, j, physicsClientId=c)
                     for j in self._joint_idx]
        jpos      = np.array([s[0] for s in js])
        jvel      = np.array([s[1] for s in js])
        return np.concatenate([pos, euler, lv, av, jpos, jvel])

    def _reward(self, action: np.ndarray, obs: np.ndarray) -> float:
        c = self._client
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=c)
        _, av    = p.getBaseVelocity(self._robot, physicsClientId=c)
        euler    = p.getEulerFromQuaternion(orn)

        dt       = 1.0 / 240.0
        fwd_vel  = (pos[0] - self._prev_x) / dt
        self._prev_x = pos[0]

        tilt        = abs(euler[0]) + abs(euler[1])
        angvel_pen  = float(np.linalg.norm(av[:2]))
        energy      = float(np.sum(action ** 2))

        return (
            1.5  * fwd_vel
            + 0.3                      # alive bonus
            - 0.5  * tilt
            - 0.1  * angvel_pen
            - 0.005 * energy
        )

    def _done(self, obs: np.ndarray) -> bool:
        z     = obs[2]
        roll  = abs(obs[3])
        pitch = abs(obs[4])
        return (
            z < self.MIN_HEIGHT or z > self.MAX_HEIGHT
            or roll  > self.MAX_TILT
            or pitch > self.MAX_TILT
            or self._step_n >= self.MAX_STEPS
        )


# ══════════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self, n: int, obs_dim: int, act_dim: int):
        self.obs     = np.empty((n, obs_dim), np.float32)
        self.actions = np.empty((n, act_dim), np.float32)
        self.rewards = np.empty(n,            np.float32)
        self.logps   = np.empty(n,            np.float32)
        self.values  = np.empty(n,            np.float32)
        self.dones   = np.empty(n,            bool)
        self._n = n
        self._i = 0

    def add(self, obs, act, rew, logp, val, done):
        i = self._i
        self.obs[i], self.actions[i], self.rewards[i] = obs, act, rew
        self.logps[i], self.values[i], self.dones[i]  = logp, val, done
        self._i += 1

    def full(self) -> bool:
        return self._i == self._n

    def reset(self):
        self._i = 0

    def compute_gae(self, last_val: float, gamma=0.99, lam=0.95):
        """GAE-Lambda returns and normalised advantages."""
        advs = np.empty(self._n, np.float32)
        gae  = 0.0
        for t in reversed(range(self._n)):
            nv   = self.values[t + 1] if t < self._n - 1 else last_val
            nd   = float(self.dones[t])
            delta = self.rewards[t] + gamma * nv * (1 - nd) - self.values[t]
            gae   = delta + gamma * lam * (1 - nd) * gae
            advs[t] = gae
        returns = advs + self.values
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs, returns


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    env    = SpiderBotEnv(render=args.render)
    ac     = ActorCritic(SpiderBotEnv.OBS_DIM, SpiderBotEnv.ACT_DIM)
    a_opt  = Adam(lr=args.lr)
    c_opt  = Adam(lr=args.lr)
    buf    = RolloutBuffer(args.n_steps, SpiderBotEnv.OBS_DIM, SpiderBotEnv.ACT_DIM)

    ep_rewards: deque = deque(maxlen=100)
    obs, _   = env.reset()
    ep_r     = 0.0
    update   = 0
    total    = 0
    ckpt_dir = args.save_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    next_save = args.save_every

    print(f"\n{'='*62}")
    print(f"  SpiderBot PPO  |  obs={SpiderBotEnv.OBS_DIM}  act={SpiderBotEnv.ACT_DIM}")
    print(f"  total_steps={args.steps:,}  n_steps={args.n_steps}  lr={args.lr}")
    print(f"{'='*62}\n")

    t0 = time.time()

    while total < args.steps:

        # ── collect rollout ──────────────────────────────────────────────
        buf.reset()
        while not buf.full():
            act, logp = ac.act(obs[None])
            act, logp = act[0], float(logp[0])
            val       = float(ac.value(obs[None])[0])

            obs_next, rew, done, _, _ = env.step(act)
            ep_r  += rew
            total += 1

            buf.add(obs, act, rew, logp, val, done)
            obs = obs_next

            if done:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                obs, _ = env.reset()

        last_val = float(ac.value(obs[None])[0])
        advs, rets = buf.compute_gae(last_val, args.gamma, args.lam)

        # ── PPO epochs ────────────────────────────────────────────────────
        update   += 1
        pl = vl = ent = 0.0
        nb = 0
        for _ in range(args.n_epochs):
            idx = np.random.permutation(args.n_steps)
            for s in range(0, args.n_steps, args.batch_size):
                b = idx[s: s + args.batch_size]
                if len(b) < 8:
                    continue
                _pl, _vl, _ent = ac.ppo_update(
                    buf.obs[b], buf.actions[b], buf.logps[b],
                    advs[b], rets[b],
                    a_opt, c_opt,
                    clip_eps=args.clip_eps,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    max_grad=args.max_grad,
                )
                pl += _pl; vl += _vl; ent += _ent
                nb += 1

        if nb:
            pl /= nb; vl /= nb; ent /= nb

        # ── logging ───────────────────────────────────────────────────────
        if update % 5 == 0:
            mean_r = np.mean(ep_rewards) if ep_rewards else float("nan")
            elapsed = time.time() - t0
            sps     = total / elapsed
            print(
                f"  step {total:>8,} | upd {update:>4} | "
                f"ep_r {mean_r:>8.2f} | "
                f"pol {pl:>7.4f} | val {vl:>7.4f} | ent {ent:>6.3f} | "
                f"{sps:.0f} sps"
            )

        # ── checkpoint ───────────────────────────────────────────────────
        if total >= next_save:
            path = os.path.join(ckpt_dir, f"spiderbot_{total:08d}.pkl")
            ac.save(path, extra={"total_steps": total, "update": update})
            ac.save(os.path.join(ckpt_dir, "latest.pkl"),
                    extra={"total_steps": total})
            next_save += args.save_every

    ac.save(os.path.join(ckpt_dir, "final.pkl"), extra={"total_steps": total})
    env.close()
    print(f"\nTraining complete — {total:,} steps in {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    ac  = ActorCritic(SpiderBotEnv.OBS_DIM, SpiderBotEnv.ACT_DIM)
    meta = ac.load(args.eval)
    print(f"Loaded {args.eval}  {meta}")

    env = SpiderBotEnv(render=True)
    for ep in range(args.eval_eps):
        obs, _ = env.reset()
        ep_r, steps, done = 0.0, 0, False
        while not done:
            act, _ = ac.act(obs[None], deterministic=True)
            obs, r, done, _, _ = env.step(act[0])
            ep_r += r
            steps += 1
        print(f"  Episode {ep + 1:>2}: reward={ep_r:8.2f}  steps={steps}")
    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="SpiderBot PPO (NumPy)")
    ap.add_argument("--render",     action="store_true")
    ap.add_argument("--eval",       type=str, default=None, metavar="CKPT")
    ap.add_argument("--eval-eps",   type=int,   default=5)
    ap.add_argument("--steps",      type=int,   default=2_000_000)
    ap.add_argument("--n-steps",    type=int,   default=2048)
    ap.add_argument("--n-epochs",   type=int,   default=10)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--gamma",      type=float, default=0.99)
    ap.add_argument("--lam",        type=float, default=0.95)
    ap.add_argument("--clip-eps",   type=float, default=0.2)
    ap.add_argument("--vf-coef",    type=float, default=0.5)
    ap.add_argument("--ent-coef",   type=float, default=0.01)
    ap.add_argument("--max-grad",   type=float, default=0.5)
    ap.add_argument("--save-dir",   type=str,   default="checkpoints")
    ap.add_argument("--save-every", type=int,   default=100_000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
