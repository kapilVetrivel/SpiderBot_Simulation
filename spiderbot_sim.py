# Create a Robot Simulation using Pybullet
import pybullet as p
import time
import pybullet_data
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Get current working directory
cwd = os.getcwd()
robot_urdf = os.path.join(cwd, 'spiderbot_assy/spiderbot_assy.urdf')
render = True  # Set to True to enable rendering, False for headless mode

class robot_sim():

    ## Iniitalize variables
    def __init__(self):
        self.robot_urdf = robot_urdf
        self.render = render
        print(self.robot_urdf)

        self.run_sim()

    ## Load Pybullet environment
    def load_environment(self):
        self.physics_client = p.connect(p.GUI if self.render else p.DIRECT)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)


    ## Load plane and set gravity
    def setup_physics(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81*1)

    ## Load robot URDF
    def load_robot(self):
        # Define flags
        URDF_USE_INERTIA_FROM_FILE = p.URDF_USE_INERTIA_FROM_FILE
        URDF_USE_SELF_COLLISION = p.URDF_USE_SELF_COLLISION
        self.flags = URDF_USE_INERTIA_FROM_FILE | URDF_USE_SELF_COLLISION


        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_urdf, start_pos, start_orientation, flags=self.flags)

    ## Get robot position and orientation
    def get_robot_state(self):
        # return position and orientation of the robot base
        return p.getBasePositionAndOrientation(self.robot_id)
    
    ## Get No. of joints and joint info
    def get_robot_joints(self):
        num_joints = p.getNumJoints(self.robot_id)
        joint_info = [p.getJointInfo(self.robot_id, i) for i in range(num_joints)]
        return num_joints, joint_info
    
    ## Set camera to top-down view fitted to the robot body
    def setup_camera(self):
        pos, _ = self.get_robot_state()
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,   # adjust to zoom in/out
            cameraYaw=0,
            cameraPitch=0,      # -90 = straight down (top view)
            cameraTargetPosition=pos
        )

    ## Initialize robot in the environment
    def initialize_robot(self, duration=5):
        print("--- --> Initializing Robot in Environment...")
        self.time = 0
        while self.time < duration:
            p.stepSimulation()
            time.sleep(1./240.)
            self.time += 1./240.
        print("--- --> Robot Initialized.")

    ## Move joints to specified positions using setJointMotorControlArray & Positional Control
    ## by receiving incremental target positions for each joint in the form of a list
    def move_joints(self, joint_indices, incremental_positions):

        num_joints, joint_info = self.get_robot_joints()
        current_positions = [p.getJointState(self.robot_id, i)[0] for i in range(num_joints)]
        print(f"Current Joint Positions: {current_positions}")
        input(f"Press Enter to move joints by {incremental_positions}...")
        target_positions = [current_positions[i] + incremental_positions[i] for i in joint_indices]

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions
        )

        # Step simulation to move joints
        for _ in range(240):  # simulate for 1 second at 240Hz
            p.stepSimulation()
            time.sleep(1./240.)

    ## Disconnect and Close Pybullet    def close_sim(self):
    def close_environment(self):
        p.disconnect()

    ## Run Sim
    def run_sim(self):
        print("\n--> Loading Environment...")
        self.load_environment()

        print("\n--> Setting up Physics...")
        self.setup_physics()

        print("\n--> Loading Robot...")
        self.load_robot()

        print("\n--> Setting up Camera...")
        self.setup_camera()

        print("\n--> Starting Simulation. Press Ctrl+C to stop.")
        try:
            # Run the simulation for a few seconds to initialize the robot and environment
            self.initialize_robot(2)

            # move all joints by +5deg & -5deg twice to test joint control
            num_joints, joint_info = self.get_robot_joints()
            joint_indices = list(range(num_joints))
            incremental_positions = [0.0873] * num_joints  # +5 degrees
            self.move_joints(joint_indices, incremental_positions)
            incremental_positions = [-0.0873] * num_joints  # -5 degrees
            self.move_joints(joint_indices, incremental_positions)

            input("\n--> Simulation Running. Press Enter to stop...")

        except KeyboardInterrupt:
            print("\n--> Simulation Interrupted by User.")

        except Exception as e:
            print("\n--> Error Occurred: ", e)

        print("\n--> Closing Environment")
        self.close_environment()



def main():
    clear_console()
    try:
        robot_sim()
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")



if __name__ == "__main__":
    # Run the main function
    main()