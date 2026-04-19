# Create a Robot Simulation using Pybullet
import pybullet as p
import time, math
import pybullet_data
import os
import traceback

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
        p.setRealTimeSimulation(1)  # Disable real-time simulation for better control over timing
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


        start_pos = [0, 0, 1]
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
        
        # create a list of joint indices, name, and angle limits
        self.joint_indices = []
        for info in joint_info:
            joint_index = info[0]
            joint_name = info[1].decode('utf-8')
            lower_limit = round(info[8],2)
            upper_limit = round(info[9],2)
            self.joint_indices.append((joint_index, joint_name, lower_limit, upper_limit))
            print(f"Joint Index: {joint_index}, Name: {joint_name}, Limits: [{lower_limit}, {upper_limit}] rads")
    
    ## Set camera to top-down view fitted to the robot body
    def setup_camera(self):
        pos, _ = self.get_robot_state()
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,   # adjust to zoom in/out
            cameraYaw=45,
            cameraPitch=-90,      # -90 = straight down (top view)
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

    ## Sweep all joints through their range of motion to test for collisions and joint limits
    ## One leg at a time, sweep each joint from its lower limit to upper limit and return to midpoint position
    def test_joint_sweep(self, duration=1):
        print("--- --> Testing Joint Sweep...")
        for joint_index, joint_name, lower_limit, upper_limit in self.joint_indices:
            print(f"Testing Joint: {joint_name} (Index: {joint_index})")
            midpoint = (lower_limit + upper_limit) / 2
            p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL, targetPosition=midpoint)
            self.time = 0
            while self.time < duration:
                p.stepSimulation()
                time.sleep(1./2.)
                self.time += 1./2.
        print("--- --> Joint Sweep Test Completed.")

    ## Disconnect and Close Pybullet    def close_sim(self):
    def close_environment(self):
        p.disconnect()

    ## Run Sim
    def run_sim(self):
        print("\n--> Loading Environment...")
        self.load_environment()

        print("\n--> Setting up Physics...")
        self.setup_physics()

        print("\n--> Loading Robot and getting Joint Info...")
        self.load_robot()
        self.get_robot_joints()

        print("\n--> Setting up Camera...")
        self.setup_camera()

        print("\n--> Starting Simulation. Press Ctrl+C to stop.")
        try:
            # Run the simulation for a few seconds to initialize the robot and environment
            self.initialize_robot(2)   

            # Sweep all joints through their range of motion to test for collisions and joint limits
            self.test_joint_sweep(1)

            input("\n--> Simulation Running. Press Enter to stop...")

        except KeyboardInterrupt:
            print("\n--> Simulation Interrupted by User.")

        except Exception as e:
            print("\n--> Error Occurred: ", e)
            print(traceback.format_exc())

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