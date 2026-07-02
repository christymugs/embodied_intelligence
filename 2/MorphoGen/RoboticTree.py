# MorphoGen/RoboticTree.py
import uuid

class Site:
    def __init__(self, Name, Pos, Euler=None, Index=None):
        self.Name = Name
        self.Pos = Pos
        self.Euler = Euler
        self.Index = Index

class Joint:
    def __init__(self, Name, JointType, Axis, RangeMin, RangeMax):
        self.Name = Name
        self.JointType = JointType
        self.Axis = Axis
        self.RangeMin = RangeMin
        self.RangeMax = RangeMax
        self.Damping = 10.0

class Actuator:
    def __init__(self, Name, ActuatorType, TargetJoint, CtrlRange=None, Gear=1.0):
        self.Name = Name
        self.ActuatorType = ActuatorType
        self.TargetJoint = TargetJoint
        self.CtrlRange = CtrlRange
        self.Gear = Gear

class Limb:
    def __init__(self, ParentBody, Name, Radius, Height, Density, LimbID=None):
        self.ParentBody = ParentBody
        self.Name = Name
        self.LimbID = LimbID
        self.Radius = Radius
        self.Height = Height
        self.Density = Density
        self.Sites = []
        self.Connections = []
        self._NextSiteIndex = 0

    def AddSite(self, Pos, Euler=None) -> Site:
        Idx = self._NextSiteIndex
        self._NextSiteIndex += 1
        Name = f"{self.Name}_Site{Idx}"
        NewSite = Site(Name, Pos, Euler, Index=Idx)
        self.Sites.append(NewSite)
        return NewSite

class Connection:
    def __init__(self, ParentLimb, ParentSite, ChildLimb):
        self.ParentLimb = ParentLimb
        self.ChildLimb = ChildLimb
        self.ParentSite = ParentSite
        self.Joints = []
        self.Actuators = []
        self._NextJointIndex = 0

    def NextJointIndex(self):
        Idx = self._NextJointIndex
        self._NextJointIndex += 1
        return Idx

class RobotTree:
    def __init__(self, RootLimb):
        self.UID = str(uuid.uuid4())[:8]
        self._NextLimbID = 0
        self.Root = self._InitRoot(RootLimb)

    def _InitRoot(self, RootLimb) -> Limb:
        if RootLimb.LimbID is None:
            RootLimb.LimbID = self._NextLimbID
            self._NextLimbID += 1
        if RootLimb.Name is None:
            RootLimb.Name = f"Limb{RootLimb.LimbID}_{self.UID}"
        return RootLimb

    def _MakeLimbNameAndID(self):
        LimbID = self._NextLimbID
        self._NextLimbID += 1
        return f"Limb{LimbID}_{self.UID}", LimbID

    def AddLimb(self, ParentLimb, ParentSite, Radius, Height, Density):
        Name, LimbID = self._MakeLimbNameAndID()
        ChildLimb = Limb(ParentLimb, Name, Radius, Height, Density, LimbID)
        Conn = Connection(ParentLimb, ParentSite, ChildLimb)
        ParentLimb.Connections.append(Conn)
        return ChildLimb, Conn

    def AddJointToConnection(self, Connection, JointType, Axis, RangeMin, RangeMax) -> Joint:
        Name = f"Joint_{Connection.ParentLimb.Name}_{Connection.ChildLimb.Name}_{Connection.NextJointIndex()}"
        NewJoint = Joint(Name, JointType, Axis, RangeMin, RangeMax)
        Connection.Joints.append(NewJoint)
        return NewJoint

    def AddActuatorToConnection(self, Connection, TargetJoint, ActuatorType="position", Gear=1.0, CtrlRange=None) -> Actuator:
        Name = f"{ActuatorType}_Actuator_{TargetJoint.Name}"
        NewActuator = Actuator(Name, ActuatorType, TargetJoint, CtrlRange, Gear)
        Connection.Actuators.append(NewActuator)
        return NewActuator