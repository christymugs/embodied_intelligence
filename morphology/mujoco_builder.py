"""Compile a RobotTree into MuJoCo MJCF XML.

Three fixes vs the original draft are marked with `# FIX:` below.

Limbs don't collide with each other by contype/conaffinity (see the FIX in
_emit_body_with_geom) -- except a limb and its OWN direct parent, which are
force-included via explicit <contact><pair> elements in build_etree, mimicking
a real hinge's mechanical stop. A first attempt at this reopened the original
jamming exploit (~14,500 static force at rest); root cause was genome.py's
attach offset only clearing the PARENT's radius, not the CHILD's own -- the
child's rounded end-cap was guaranteed to overlap the parent by construction,
on every attachment, before evolution ever touched anything. Fixed at the
source (genome.py now offsets by parent radius + child radius, true tangency),
so this pair mechanism is safe to re-enable on top of that.
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
        self._emit_parent_child_contact_pairs(mj_root, root_limb)
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
            # FIX: contype/conaffinity pair with the limb geoms below — floor
            # collides with every limb, limbs never collide with each other.
            contype="2",
            conaffinity="1",
        )

    def _emit_body_with_geom(self, limb: Limb, is_root: bool = False) -> etree._Element:
        body_name = limb.name or "unnamed"
        body = etree.Element("body", name=body_name)

        # Vertical capsule along +z, from (0,0,0) to (0,0,height)
        fromto = f"0 0 0 0 0 {limb.height}"
        # Torso rendered in a distinct color from every limb -- with several
        # gray-on-gray segments and no bracket/flange geometry to read as
        # "mounted," it was hard to tell which piece was the root body at a
        # glance in the viewer. Purely cosmetic: no effect on physics.
        rgba = "0.85 0.25 0.2 1" if is_root else "0.5 0.5 0.55 1"
        etree.SubElement(
            body, "geom", name=f"{body_name}_geom", type="capsule", fromto=fromto,
            size=str(limb.radius), density=str(limb.density), rgba=rgba,
            # FIX: without an explicit contype/conaffinity, MuJoCo's default lets
            # any two non-adjacent limb geoms collide. Nothing in genome/mutation
            # stops evolution from placing limbs so they overlap, so the search
            # was finding morphologies that jam limbs into each other and ride
            # the constraint solver's resolution force instead of learning to
            # walk. Limbs only collide with the floor (contype=2/conaffinity=1
            # above), never with each other.
            contype="1",
            conaffinity="2",
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

    def _emit_parent_child_contact_pairs(self, mj_root: etree._Element, root_limb: Limb) -> None:
        """Force-include collision between each limb and its own direct
        parent only (see module docstring) -- MuJoCo excludes adjacent
        (parent/child) body pairs from collision by default, same as it
        excludes any pair contype/conaffinity doesn't grant, so this needs an
        explicit <pair> either way; a bare `name="..._geom"` reference is
        enough, no friction/solver overrides."""
        pairs: list[tuple[str, str]] = []
        self._collect_parent_child_pairs(root_limb, pairs)
        if not pairs:
            return

        contact_root = mj_root.find("contact")
        if contact_root is None:
            contact_root = etree.SubElement(mj_root, "contact")
        for geom1, geom2 in pairs:
            etree.SubElement(contact_root, "pair", geom1=geom1, geom2=geom2)

    def _collect_parent_child_pairs(self, limb: Limb, out_list: list[tuple[str, str]]) -> None:
        for conn in limb.connections:
            out_list.append((f"{limb.name}_geom", f"{conn.child_limb.name}_geom"))
            self._collect_parent_child_pairs(conn.child_limb, out_list)

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

        if act.forcerange is not None:
            # Caps the actuator's own force before the gear (mechanical
            # advantage) multiplies it through to the joint -- see
            # morphology/genome.py:_actuator_force_cap.
            attrs["forcerange"] = f"{act.forcerange[0]} {act.forcerange[1]}"
            attrs["forcelimited"] = "true"

        # FIX: was `is "position"` — string identity is not guaranteed; use ==.
        if act.actuator_type == "position":
            attrs["kp"] = "100.0"  # default proportional gain

        etree.SubElement(actuator_root, act.actuator_type, **attrs)

    def _build_limb_subtree(self, limb: Limb, is_root: bool = False) -> etree._Element:
        body_elem = self._emit_body_with_geom(limb, is_root=is_root)
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
            if conn.parent_site.quat is not None:
                qw, qx, qy, qz = conn.parent_site.quat
                child_body.set("quat", f"{qw} {qx} {qy} {qz}")
            elif conn.parent_site.euler is not None:
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
            if site.quat is not None:
                qw, qx, qy, qz = site.quat
                attrs["quat"] = f"{qw} {qx} {qy} {qz}"
            elif site.euler is not None:
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