from RoboticTree import *
from MujocoBuilder import MujocoBuilder

if __name__ == "__main__":

    # Create root limb
    root = Limb(
        parent_body=None,
        name=None,  # let RobotTree auto-name it limb0
        radius=0.05,
        height=0.5,
        density=1000,
    )
 
    tree = RobotTree(root)

    # Add a site at the end of the root limb
    root_site = root.add_site(pos=(0, 0, root.height))

    # First we add the limb + connection
    child, conn = tree.add_limb(
        parent_limb=root,
        parent_site=root_site,
        radius=0.04,
        height=0.4,
        density=800,
    )

    j = tree.add_joint_to_connection(
        conn,
        joint_type="hinge",
        axis=[0, 1, 0],
        range_min=-90,
        range_max=90,
    )

    #  Attach an actuator to that joint
    act = tree.add_actuator_to_connection(
        conn,
        target_joint=j,
        actuator_type="position",  # or "motor" if you change the builder
        gear=1.0,
    )

    child2, conn2 = tree.add_limb(
        parent_limb=child,
        parent_site=child.add_site(pos=(0, 0, child.height)),
        radius=0.03,
        height=0.3,
        density=600,
    )

    j2 = tree.add_joint_to_connection(
        conn2,
        joint_type="hinge",
        axis=[1, 0, 0],
        range_min=-45,
        range_max=45,
    )

    act2 = tree.add_actuator_to_connection(
        conn2,
        target_joint=j2,
        actuator_type="position",
    )

    child3, conn3 = tree.add_limb(
        parent_limb=child2,
        parent_site=child2.add_site(pos=(0, 0, child2.height)),
        radius=0.02,
        height=0.2,
        density=400,
    )

    j3 = tree.add_joint_to_connection(
        conn3,
        joint_type="hinge",
        axis=[1, 0, 0],
        range_min=-30,
        range_max=30,
    )

    act3 = tree.add_actuator_to_connection(
        conn3,
        target_joint=j3,
        actuator_type="position",
    )

    #  Build and save MuJoCo XML
    builder = MujocoBuilder(model_name="test_robot")
    builder.save(tree, "test_robot.xml")

    print("Wrote test_robot.xml")
