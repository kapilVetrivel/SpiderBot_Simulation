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
render = True

class robot_sim():

    ## Iniitalize variables
    def __init__(self):
        self.robot_urdf = robot_urdf
        self.render = render
        print(self.robot_urdf)

        self.run_sim()

    ## Load Pybullet environment
    def load_environment(self):
        physics_client = p.connect(p.GUI if self.render else p.DIRECT)

    ## Load plane and set gravity
    def setup_physics(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

    ## Load robot URDF
    def load_robot(self):
        start_pos = [0, 0, 0.1]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_urdf, start_pos, start_orientation)

    ## Disconnect and Close Pybullet    def close_sim(self):
    def close_environment():
        p.disconnect()

    ## Run Sim
    def run_sim(self):
        print("\n--> Loading Environment...")
        self.load_environment()

        print("\n--> Setting up Physics...")
        self.setup_physics()

        print("\n--> Loading Robot...")
        self.load_robot()

        while True:
            p.stepSimulation()
            time.sleep(1./240.)

        print("\n--> Closing Environment")
        self.close_environment()



def main():
    try:
        robot_sim()
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")



if __name__ == "__main__":
    # Run the main function
    main()