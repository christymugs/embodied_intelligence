# Tests/TestBasicMorphology.py
from MorphoGen.RoboticTree import RobotTree, Limb
from Simulation.MujocoBuilder import MujocoBuilder

def run_test():
    # Create root limb
    root = Limb(parent_body=None, name="limb0", radius=0.05, height=0.5, density=1000)
    tree = RobotTree(root)

    # Add a site at the end of the root limb
    root_site = root.add_site(pos=(0, 0, root.height))

    # Add limbs and connections
    child, conn = tree.add_limb(root, root_site, 0.04, 0.4, 800)
    j = tree.add_joint_to_connection(conn, "hinge", [0, 1, 0], -90, 90)
    tree.add_actuator_to_connection(conn, j, "position")

    # Build and save
    builder = MujocoBuilder(model_name="test_robot")
    try:
        builder.save(tree, "test_robot.xml")
        print("Success: test_robot.xml generated.")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    run_test()