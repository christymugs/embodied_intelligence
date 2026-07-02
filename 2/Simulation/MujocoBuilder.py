# Simulation/MujocoBuilder.py
from lxml import etree
from MorphoGen.RoboticTree import RobotTree, Limb

class MujocoBuilder:
    def __init__(self, ModelName="RobotModel"):
        self.ModelName = ModelName

    def BuildETree(self, RobotTree: RobotTree) -> etree._ElementTree:
        MjRoot, WorldBody = self._MakeMujocoRoot()
        self._AddFloor(WorldBody)
        RootBody = self._BuildLimbSubtree(RobotTree.Root, IsRoot=True)
        WorldBody.append(RootBody)
        self._EmitAllActuators(MjRoot, RobotTree.Root)
        return etree.ElementTree(MjRoot)

    def Save(self, RobotTree: RobotTree, Path: str) -> None:
        Tree = self.BuildETree(RobotTree)
        Tree.write(Path, xml_declaration=True, encoding="utf-8", pretty_print=True)

    def _MakeMujocoRoot(self):
        Root = etree.Element("mujoco", model=self.ModelName)
        etree.SubElement(Root, "option", gravity="0 0 -9.81", timestep="0.002", 
                         solver="CG", iterations="500", integrator="Euler")
        WorldBody = etree.SubElement(Root, "worldbody")
        return Root, WorldBody

    def _AddFloor(self, WorldBody):
        etree.SubElement(WorldBody, "geom", name="Floor", type="plane", 
                         size="10 10 0.1", pos="0 0 0", rgba="0.8 0.8 0.8 1")

    def _BuildLimbSubtree(self, Limb: Limb, IsRoot: bool = False) -> etree._Element:
        Body = etree.Element("body", name=Limb.Name)
        if IsRoot: Body.set("pos", "0 0 0.5")
        
        # Physics Stability: contype/conaffinity set to 1 for floor interaction
        etree.SubElement(Body, "geom", type="capsule", fromto=f"0 0 0 0 0 {Limb.Height}", 
                         size=str(Limb.Radius), density=str(Limb.Density), 
                         contype="1", conaffinity="1")
        
        for Conn in Limb.Connections:
            ChildBody = self._BuildLimbSubtree(Conn.ChildLimb)
            Sx, Sy, Sz = Conn.ParentSite.Pos
            ChildBody.set("pos", f"{Sx} {Sy} {Sz}")
            
            for Joint in Conn.Joints:
                etree.SubElement(ChildBody, "joint", name=Joint.Name, type=Joint.JointType, 
                                 axis=" ".join(map(str, Joint.Axis)), 
                                 range=f"{Joint.RangeMin} {Joint.RangeMax}")
            
            Body.append(ChildBody)
        return Body

    def _EmitAllActuators(self, MjRoot, RootLimb):
        ActRoot = etree.SubElement(MjRoot, "actuator")
        def Collect(Limb):
            for Conn in Limb.Connections:
                for Act in Conn.Actuators:
                    etree.SubElement(ActRoot, Act.ActuatorType, name=Act.Name, 
                                     joint=Act.TargetJoint.Name, gear=str(Act.Gear))
                Collect(Conn.ChildLimb)
        Collect(RootLimb)