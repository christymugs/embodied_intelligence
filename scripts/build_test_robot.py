"""Build the original hand-designed 3-limb robot and save its MJCF.

    python -m scripts.build_test_robot
"""

from morphology.robotic_tree import Limb, RobotTree
from morphology.mujoco_builder import MujocoBuilder


def main():
    root = Limb(parent_body=None, name=None, radius=0.05, height=0.5, density=1000)
    tree = RobotTree(root)
    root_site = root.add_site(pos=(0, 0, root.height))

    child, conn = tree.add_limb(root, root_site, radius=0.04, height=0.4, density=800)
    j = tree.add_joint_to_connection(conn, "hinge", [0, 1, 0], -90, 90)
    tree.add_actuator_to_connection(conn, j, actuator_type="position", gear=1.0)

    child2, conn2 = tree.add_limb(
        child, child.add_site(pos=(0, 0, child.height)),
        radius=0.03, height=0.3, density=600,
    )
    j2 = tree.add_joint_to_connection(conn2, "hinge", [1, 0, 0], -45, 45)
    tree.add_actuator_to_connection(conn2, j2, actuator_type="position")

    child3, conn3 = tree.add_limb(
        child2, child2.add_site(pos=(0, 0, child2.height)),
        radius=0.02, height=0.2, density=400,
    )
    j3 = tree.add_joint_to_connection(conn3, "hinge", [1, 0, 0], -30, 30)
    tree.add_actuator_to_connection(conn3, j3, actuator_type="position")

    builder = MujocoBuilder(model_name="test_robot")
    builder.save(tree, "experiments/test_robot.xml")
    print("Wrote experiments/test_robot.xml")


if __name__ == "__main__":
    main()