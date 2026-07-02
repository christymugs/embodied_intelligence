"""Compile a RobotTree into MuJoCo MJCF XML.

Two fixes vs the original draft are marked with `# FIX:` below.
"""

from lxml import etree

from .robotic_tree import Limb, Joint, Actuator, RobotTree, Connection


class MujocoBuilder:
    def __init__(self, model_name="robot_model"):
        self.model_name = model_name

    def build_etree(self, robot_tree: RobotTree) -> etree._ElementTree:
        """Build and return an etree.ElementTree for the given RobotTree."""
        mj_root, worldbody = self._make_mujoco_root()

        self._add_light(worldbody)
        floor_elem = self._make_floor()
        worldbody.append(floor_elem)

        root_limb = robot_tree.root
        root_body = self._build_limb_subtree(root_limb, is_root=True)
        worldbody.append(root_body)

        self._emit_all_actuators(mj_root, root_limb)
        return etree.ElementTree(mj_root)

    def to_xml_string(self, robot_tree: RobotTree) -> str:
        """Return the MJCF as a string (handy for loading from memory)."""
        tree = self.build_etree(robot_tree)
        return etree.tostring(tree, pretty_print=True, encoding="unicode")

    def etree_from_xml(self, xml, ispath=True):
        if ispath:
            parser = etree.XMLParser(remove_blank_text=True)
            root = etree.parse(xml, parser).getroot()
        else:
            root = etree.fromstring(xml)
        tree = etree.ElementTree(root)
        return root, tree

    def save(self, robot_tree: RobotTree, path: str) -> None:
        tree = self.build_etree(robot_tree)
        tree.write(path, xml_declaration=True, encoding="utf-8", pretty_print=True)

    # ###########################################################
    # ##################### INTERNAL HELPERS ####################
    # ###########################################################

    def _make_mujoco_root(self):
        root = etree.Element("mujoco", model=self.model_name)
        worldbody = etree.SubElement(root, "worldbody")
        return root, worldbody

    def _add_light(
        self,
        worldbody,
        name="light",
        pos="0 -3 10",
        dir="0 0.5 -1",
        diffuse=".8 .8 .8",
        specular="0.1 0.1 0.1",
    ):
        return etree.SubElement(
            worldbody, "light", name=name, pos=pos, dir=dir,
            diffuse=diffuse, specular=specular,
        )

    def _make_floor(self, size=10.0, thickness=0.1):
        # FIX: dropped material="" — an empty material ref makes MuJoCo warn /
        # error on compile. Omit the attribute entirely instead.
        return etree.Element(
            "geom",
            name="floor",
            type="plane",
            size=f"{size} {size} {thickness}",
            pos="0 0 0",
            rgba="0.8 0.8 1 1",
        )

    def _emit_body_with_geom(self, limb: Limb) -> etree._Element:
        body_name = limb.name or "unnamed"
        body = etree.Element("body", name=body_name)

        # Vertical capsule along +z, from (0,0,0) to (0,0,height)
        fromto = f"0 0 0 0 0 {limb.height}"
        etree.SubElement(
            body, "geom", type="capsule", fromto=fromto,
            size=str(limb.radius), density=str(limb.density),
        )
        return body

    def _emit_all_actuators(self, mj_root: etree._Element, root_limb: Limb) -> None:
        actuators: list[Actuator] = []
        self._collect_actuators(root_limb, actuators)
        if not actuators:
            return

        act_root = mj_root.find("actuator")
        if act_root is None:
            act_root = etree.SubElement(mj_root, "actuator")
        for act in actuators:
            self._emit_single_actuator(act_root, act)

    def _collect_actuators(self, limb: Limb, out_list: list[Actuator]) -> None:
        for conn in limb.connections:
            out_list.extend(conn.actuators)
            self._collect_actuators(conn.child_limb, out_list)

    def _emit_single_actuator(self, actuator_root: etree.Element, act: Actuator) -> None:
        assert act.target_joint is not None, "Actuator must have a target_joint."

        attrs = {
            "name": act.name,
            "joint": act.target_joint.name,
            "gear": str(act.gear),
        }
        if act.ctrlrange is None:
            attrs["inheritrange"] = "1"
        else:
            attrs["ctrlrange"] = f"{act.ctrlrange[0]} {act.ctrlrange[1]}"

        # FIX: was `is "position"` — string identity is not guaranteed; use ==.
        if act.actuator_type == "position":
            attrs["kp"] = "10.0"  # default proportional gain

        etree.SubElement(actuator_root, act.actuator_type, **attrs)

    def _build_limb_subtree(self, limb: Limb, is_root: bool = False) -> etree._Element:
        body_elem = self._emit_body_with_geom(limb)
        self._emit_sites(body_elem, limb)

        if is_root:
            body_elem.set("pos", "0 0 0")
            # A free joint lets the whole robot move/fall under gravity.
            # Without it the root is welded to the world and nothing locomotes.
            etree.SubElement(body_elem, "freejoint", name=f"{limb.name}_root")

        for conn in limb.connections:
            child_body = self._build_limb_subtree(conn.child_limb, is_root=False)

            sx, sy, sz = conn.parent_site.pos
            child_body.set("pos", f"{sx} {sy} {sz}")
            if conn.parent_site.euler is not None:
                ex, ey, ez = conn.parent_site.euler
                child_body.set("euler", f"{ex} {ey} {ez}")

            for idx, j in enumerate(conn.joints):
                self._emit_joint(child_body, j, conn, local_index=idx)

            body_elem.append(child_body)

        return body_elem

    def _emit_sites(self, body_elem: etree._Element, limb: Limb) -> None:
        for site in limb.sites:
            sx, sy, sz = site.pos
            attrs = {"name": site.name, "pos": f"{sx} {sy} {sz}"}
            if site.euler is not None:
                ex, ey, ez = site.euler
                attrs["euler"] = f"{ex} {ey} {ez}"
            etree.SubElement(body_elem, "site", type="sphere", size="0.01", **attrs)

    def _emit_joint(
        self, child_body_elem, joint: Joint, connection: Connection, local_index: int
    ) -> None:
        if not joint.joint_type or joint.joint_type.lower() == "none":
            return

        axis_str = " ".join(str(v) for v in (joint.axis or [0, 0, 0]))
        range_str = f"{joint.range_min} {joint.range_max}"
        etree.SubElement(
            child_body_elem, "joint",
            name=joint.name, type=joint.joint_type, axis=axis_str,
            range=range_str, damping=str(joint.damping),
        )