"""Tree-based morphology representation.

A robot is a tree of `Limb`s (capsules). Each `Connection` is a directed edge
from a parent limb (anchored at a `Site` on that limb) to a child limb, and may
carry `Joint`s and `Actuator`s. The whole tree compiles to MuJoCo XML via
`morphology.mujoco_builder.MujocoBuilder`.

This is the genotype's "phenotype builder": the evolutionary genome (see
morphology/genome.py) is just a recipe for assembling one of these trees.
"""


class Site:
    """The position/frame of an attachment point relative to its parent limb."""

    def __init__(self, name, pos, euler=None, index=None):
        self.name = name
        self.pos = pos
        self.euler = euler
        self.index = index  # local index within parent limb


class Joint:
    """A single joint living on a Connection (parent->child edge)."""

    def __init__(self, name, joint_type, axis, range_min, range_max):
        self.name = name
        self.joint_type = joint_type  # e.g., 'hinge', 'slide'
        self.axis = axis  # e.g., [1,0,0] for x-axis
        self.range_min = range_min
        self.range_max = range_max
        self.actuator = None
        self.damping = 5.0  # default damping value


class Actuator:
    """Drives exactly one Joint."""

    def __init__(
        self, name, actuator_type, target_joint: "Joint", ctrlrange=None, gear=1.0
    ):
        self.name = name
        self.actuator_type = actuator_type  # e.g., 'motor', 'position'
        self.target_joint = target_joint
        self.ctrlrange = ctrlrange  # e.g., (-1.0, 1.0)
        self.gear = gear


class Limb:
    """A capsule body. Carries sites (attachment points) and connections (children)."""

    def __init__(self, parent_body, name, radius, height, density, limb_id=None):
        self.parent_body = parent_body
        self.name = name
        self.limb_id = limb_id

        self.radius = radius
        self.height = height
        self.density = density

        self.sites: list[Site] = []
        self.connections: list["Connection"] = []

        self._next_site_index = 0

    def add_site(self, pos, euler=None) -> Site:
        """Create a new site on this limb with a unique index and auto name."""
        idx = self._next_site_index
        self._next_site_index += 1

        name = f"{self.name}_site{idx}"  # i.e. limbX_siteY
        site = Site(name, pos, euler, index=idx)
        self.sites.append(site)
        return site


class Connection:
    """A directed edge: parent_limb (at parent_site) -> child_limb."""

    def __init__(self, parent_limb: Limb, parent_site: Site, child_limb: Limb):
        self.parent_limb = parent_limb
        self.child_limb = child_limb
        self.parent_site = parent_site

        self.joints = []
        self.actuators = []
        # Only indexing joints (not actuators) => one actuator per joint.
        self._next_joint_index = 0

    def next_joint_index(self):
        idx = self._next_joint_index
        self._next_joint_index += 1
        return idx


class RobotTree:
    """Owns the root limb and the limb-id counter; provides assembly helpers."""

    def __init__(self, root_limb: Limb):
        self._next_limb_id = 0
        self.root = self._init_root(root_limb)

    def _init_root(self, root_limb: Limb) -> Limb:
        if root_limb.limb_id is None:
            root_limb.limb_id = self._next_limb_id
            self._next_limb_id += 1
        if root_limb.name is None:
            root_limb.name = f"limb{root_limb.limb_id}"  # e.g. limb0
        return root_limb

    def _make_limb_name_and_id(self):
        limb_id = self._next_limb_id
        self._next_limb_id += 1
        return f"limb{limb_id}", limb_id

    def add_limb(
        self,
        parent_limb: Limb,
        parent_site: Site,
        radius: float,
        height: float,
        density: float,
    ):
        """Create + attach a new child limb. Returns (child_limb, connection)."""
        name, limb_id = self._make_limb_name_and_id()

        child_limb = Limb(
            parent_body=parent_limb,
            name=name,
            radius=radius,
            height=height,
            density=density,
            limb_id=limb_id,
        )

        conn = Connection(
            parent_limb=parent_limb,
            parent_site=parent_site,
            child_limb=child_limb,
        )
        parent_limb.connections.append(conn)
        return child_limb, conn

    def add_joint_to_connection(
        self,
        connection: Connection,
        joint_type: str,
        axis: list[float],
        range_min: float,
        range_max: float,
    ) -> Joint:
        joint_name = (
            f"joint_{connection.parent_limb.name}_{connection.child_limb.name}"
            f"_{connection.next_joint_index()}"
        )
        j = Joint(
            name=joint_name,
            joint_type=joint_type,
            axis=axis,
            range_min=range_min,
            range_max=range_max,
        )
        connection.joints.append(j)
        return j

    def add_actuator_to_connection(
        self,
        connection: Connection,
        target_joint: Joint,
        actuator_type: str = "position",
        gear: float = 1.0,
        ctrlrange=None,
    ) -> Actuator:
        """Attach an actuator (1:1) to a joint on this connection."""
        act_name = f"{actuator_type}_actuator_{target_joint.name}"
        act = Actuator(
            name=act_name,
            actuator_type=actuator_type,
            target_joint=target_joint,
            ctrlrange=ctrlrange,
            gear=gear,
        )
        connection.actuators.append(act)
        return act