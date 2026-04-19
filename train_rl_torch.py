#!/usr/bin/env python3
"""
SpiderBot PPO Training — PyTorch implementation.

Usage:
    python train_rl_torch.py                        # train headless
    python train_rl_torch.py --render               # train with GUI
    python train_rl_torch.py --eval checkpoints_torch/latest.pt
    python train_rl_torch.py --steps 3000000 --lr 3e-4
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH  = os.path.join(SCRIPT_DIR, "spiderbot_assy", "spiderbot_assy.urdf")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# Actor-Critic network
# ══════════════════════════════════════════════════════════════════════════════

def _mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic.
    Actor outputs Gaussian mean; log_std is a learned parameter.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        sizes = [obs_dim] + list(hidden)

        self.actor_trunk = _mlp(sizes)
        self.actor_mean  = nn.Linear(hidden[-1], act_dim)

        self.critic_trunk = _mlp(sizes)
        self.critic_head  = nn.Linear(hidden[-1], 1)

        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # smaller gain for policy output head
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    # ── forward helpers ──

    def _policy(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_trunk(obs))
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_trunk(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic=False):
        dist   = self._policy(obs)
        action = dist.mean if deterministic else dist.sample()
        logp   = dist.log_prob(action).sum(-1)
        value  = self.get_value(obs)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist    = self._policy(obs)
        logp    = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value   = self.get_value(obs)
        return logp, entropy, value


# ══════════════════════════════════════════════════════════════════════════════
# SpiderBot Gymnasium environment  (identical to train_rl.py)
# ══════════════════════════════════════════════════════════════════════════════

class SpiderBotEnv(gym.Env):
    """
    Observation (28-dim): base_pos(3) + euler(3) + linvel(3) + angvel(3)
                          + joint_pos(8) + joint_vel(8)
    Action (8-dim):       normalised joint targets in [-1, 1]
    """

    OBS_DIM   = 28
    ACT_DIM   = 8
    MAX_STEPS = 1000
    SPAWN_Z   = 0.25
    MIN_HEIGHT = 0.07
    MAX_HEIGHT = 0.55
    MAX_TILT   = np.deg2rad(60)

    def __init__(self, render: bool = False):
        super().__init__()
        self._render  = render
        self._client  = None
        self._robot   = -1
        self._joint_idx: list       = []
        self._jlimits: np.ndarray   = np.empty((0, 2))
        self._step_n  = 0
        self._prev_x  = 0.0

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.ACT_DIM,), dtype=np.float32)

        self._connect()

    def _connect(self):
        mode = p.GUI if self._render else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        if self._render:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.15],
                physicsClientId=self._client)

    def _rebuild(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        self._robot = p.loadURDF(
            URDF_PATH, [0, 0, self.SPAWN_Z],
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._client)
        self._joint_idx, limits = [], []
        for i in range(p.getNumJoints(self._robot, physicsClientId=self._client)):
            info = p.getJointInfo(self._robot, i, physicsClientId=self._client)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_idx.append(i)
                limits.append((info[8], info[9]))
        self._jlimits = np.array(limits)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rebuild()
        for _ in range(60):
            p.stepSimulation(physicsClientId=self._client)
        self._step_n = 0
        self._prev_x = p.getBasePositionAndOrientation(
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

    def _apply(self, action: np.ndarray):
        lo, hi  = self._jlimits[:, 0], self._jlimits[:, 1]
        targets = lo + (np.clip(action, -1, 1) + 1.0) * 0.5 * (hi - lo)
        for j, ji in enumerate(self._joint_idx):
            p.setJointMotorControl2(
                self._robot, ji, p.POSITION_CONTROL,
                targetPosition=float(targets[j]),
                force=50.0,
                physicsClientId=self._client)

    def _obs(self) -> np.ndarray:
        c = self._client
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=c)
        lv,  av  = p.getBaseVelocity(self._robot, physicsClientId=c)
        euler    = p.getEulerFromQuaternion(orn)
        js       = [p.getJointState(self._robot, j, physicsClientId=c)
                    for j in self._joint_idx]
        return np.concatenate([pos, euler, lv, av,
                                [s[0] for s in js],
                                [s[1] for s in js]])

    def _reward(self, action: np.ndarray, obs: np.ndarray) -> float:
        c = self._client
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=c)
        _,   av  = p.getBaseVelocity(self._robot, physicsClientId=c)
        euler    = p.getEulerFromQuaternion(orn)

        fwd_vel = (pos[0] - self._prev_x) / (1.0 / 240.0)
        self._prev_x = pos[0]

        tilt       = abs(euler[0]) + abs(euler[1])
        angvel_pen = float(np.linalg.norm(av[:2]))
        energy     = float(np.sum(action ** 2))

        return (
            1.5  * fwd_vel
            + 0.3
            - 0.5  * tilt
            - 0.1  * angvel_pen
            - 0.005 * energy
        )

    def _done(self, obs: np.ndarray) -> bool:
        return (
            obs[2] < self.MIN_HEIGHT or obs[2] > self.MAX_HEIGHT
            or abs(obs[3]) > self.MAX_TILT
            or abs(obs[4]) > self.MAX_TILT
            or self._step_n >= self.MAX_STEPS
        )


# ══════════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self, n: int, obs_dim: int, act_dim: int, device: torch.device):
        self.device  = device
        self._n      = n
        self._obs    = torch.zeros(n, obs_dim)
        self._acts   = torch.zeros(n, act_dim)
        self._rews   = torch.zeros(n)
        self._logps  = torch.zeros(n)
        self._vals   = torch.zeros(n)
        self._dones  = torch.zeros(n)
        self._i      = 0

    def add(self, obs, act, rew, logp, val, done):
        i = self._i
        self._obs[i]   = obs
        self._acts[i]  = act
        self._rews[i]  = rew
        self._logps[i] = logp
        self._vals[i]  = val
        self._dones[i] = done
        self._i += 1

    def full(self) -> bool:
        return self._i == self._n

    def reset(self):
        self._i = 0

    def to(self, device):
        self._obs   = self._obs.to(device)
        self._acts  = self._acts.to(device)
        self._rews  = self._rews.to(device)
        self._logps = self._logps.to(device)
        self._vals  = self._vals.to(device)
        self._dones = self._dones.to(device)
        return self

    def compute_gae(self, last_val: torch.Tensor, gamma=0.99, lam=0.95):
        # Compute entirely in numpy to avoid dtype conflicts, then move to device
        last_val_np = float(last_val.item())
        vals = self._vals.cpu().numpy()
        rews = self._rews.cpu().numpy()
        dns  = self._dones.cpu().numpy()

        advs_np = np.zeros(self._n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self._n)):
            nv    = vals[t + 1] if t < self._n - 1 else last_val_np
            nd    = float(dns[t])
            delta = rews[t] + gamma * nv * (1.0 - nd) - vals[t]
            gae   = float(delta) + gamma * lam * (1.0 - nd) * gae
            advs_np[t] = gae

        advs    = torch.tensor(advs_np, dtype=torch.float32, device=self.device)
        returns = advs + self._vals.to(self.device)
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs, returns

    def get(self):
        return (self._obs.to(self.device),
                self._acts.to(self.device),
                self._logps.to(self.device),
                self._vals.to(self.device),
                self._dones.to(self.device))


# ══════════════════════════════════════════════════════════════════════════════
# PPO update
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update(ac: ActorCritic, optimizer: optim.Optimizer,
               obs, actions, old_logps, advantages, returns,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad=0.5):

    logps, entropy, values = ac.evaluate(obs, actions)
    ratio     = (logps - old_logps).exp()
    clip_r    = ratio.clamp(1 - clip_eps, 1 + clip_eps)

    policy_loss = -torch.min(ratio * advantages, clip_r * advantages).mean()
    value_loss  = (values - returns).pow(2).mean()
    entropy_loss = entropy.mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), max_grad)
    optimizer.step()

    return (policy_loss.item(), value_loss.item(), entropy_loss.item())


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    torch.manual_seed(0)
    np.random.seed(0)

    env  = SpiderBotEnv(render=args.render)
    ac   = ActorCritic(SpiderBotEnv.OBS_DIM, SpiderBotEnv.ACT_DIM).to(DEVICE)
    opt  = optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)
    buf  = RolloutBuffer(args.n_steps, SpiderBotEnv.OBS_DIM,
                         SpiderBotEnv.ACT_DIM, DEVICE)

    ep_rewards: deque = deque(maxlen=100)
    obs_np, _  = env.reset()
    ep_r       = 0.0
    total      = 0
    update     = 0
    next_save  = args.save_every
    ckpt_dir   = args.save_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SpiderBot PPO (PyTorch {torch.__version__})  device={DEVICE}")
    print(f"  obs={SpiderBotEnv.OBS_DIM}  act={SpiderBotEnv.ACT_DIM}  "
          f"total_steps={args.steps:,}  n_steps={args.n_steps}  lr={args.lr}")
    print(f"{'='*65}\n")
    t0 = time.time()

    while total < args.steps:

        # ── collect rollout ──────────────────────────────────────────────
        buf.reset()
        ac.eval()
        with torch.no_grad():
            while not buf.full():
                obs_t = torch.tensor(obs_np, dtype=torch.float32,
                                     device=DEVICE).unsqueeze(0)
                act_t, logp_t, val_t = ac.act(obs_t)
                act_np = act_t[0].cpu().numpy()

                obs_next, rew, done, _, _ = env.step(act_np)
                ep_r  += rew
                total += 1

                buf.add(obs_t[0].cpu(), act_t[0].cpu(),
                        rew, logp_t[0].cpu(), val_t[0].cpu(), float(done))
                obs_np = obs_next

                if done:
                    ep_rewards.append(ep_r)
                    ep_r   = 0.0
                    obs_np, _ = env.reset()

            # bootstrap last value
            last_obs = torch.tensor(obs_np, dtype=torch.float32,
                                    device=DEVICE).unsqueeze(0)
            last_val = ac.get_value(last_obs)[0]

        advs, rets = buf.compute_gae(last_val, args.gamma, args.lam)

        # ── PPO update epochs ────────────────────────────────────────────
        ac.train()
        update += 1
        obs_b, acts_b, old_logps_b, _, _ = buf.get()
        pl = vl = ent = 0.0
        nb = 0

        for _ in range(args.n_epochs):
            idx = torch.randperm(args.n_steps, device=DEVICE)
            for s in range(0, args.n_steps, args.batch_size):
                b = idx[s: s + args.batch_size]
                if len(b) < 8:
                    continue
                _pl, _vl, _ent = ppo_update(
                    ac, opt,
                    obs_b[b], acts_b[b], old_logps_b[b],
                    advs[b], rets[b],
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
            mean_r  = np.mean(ep_rewards) if ep_rewards else float("nan")
            elapsed = time.time() - t0
            print(
                f"  step {total:>8,} | upd {update:>4} | "
                f"ep_r {mean_r:>8.2f} | "
                f"pol {pl:>7.4f} | val {vl:>7.4f} | ent {ent:>6.3f} | "
                f"{total/elapsed:.0f} sps"
            )

        # ── checkpoint ────────────────────────────────────────────────────
        if total >= next_save:
            _save(ac, opt, total, update, ckpt_dir,
                  f"spiderbot_{total:08d}.pt")
            _save(ac, opt, total, update, ckpt_dir, "latest.pt")
            next_save += args.save_every

    _save(ac, opt, total, update, ckpt_dir, "final.pt")
    env.close()
    print(f"\nTraining complete — {total:,} steps in {time.time()-t0:.1f}s")


def _save(ac, opt, total_steps, update, ckpt_dir, name):
    path = os.path.join(ckpt_dir, name)
    torch.save({
        "model":      ac.state_dict(),
        "optimizer":  opt.state_dict(),
        "total_steps": total_steps,
        "update":     update,
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    ac = ActorCritic(SpiderBotEnv.OBS_DIM, SpiderBotEnv.ACT_DIM).to(DEVICE)
    ckpt = torch.load(args.eval, map_location=DEVICE, weights_only=True)
    ac.load_state_dict(ckpt["model"])
    ac.eval()
    print(f"Loaded {args.eval}  (step {ckpt.get('total_steps', '?')})")

    env = SpiderBotEnv(render=True)
    for ep in range(args.eval_eps):
        obs_np, _ = env.reset()
        ep_r, steps, done = 0.0, 0, False
        with torch.no_grad():
            while not done:
                obs_t = torch.tensor(obs_np, dtype=torch.float32,
                                     device=DEVICE).unsqueeze(0)
                act_t, _, _ = ac.act(obs_t, deterministic=True)
                obs_np, r, done, _, _ = env.step(act_t[0].cpu().numpy())
                ep_r += r
                steps += 1
        print(f"  Episode {ep + 1:>2}: reward={ep_r:8.2f}  steps={steps}")
    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="SpiderBot PPO (PyTorch)")
    ap.add_argument("--render",     action="store_true")
    ap.add_argument("--eval",       type=str,   default=None, metavar="CKPT")
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
    ap.add_argument("--save-dir",   type=str,   default="checkpoints_torch")
    ap.add_argument("--save-every", type=int,   default=100_000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
