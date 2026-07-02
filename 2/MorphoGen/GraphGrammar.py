# MorphoGen/GraphGrammar.py
import numpy as np

class MorphologyGenerator:
    def __init__(self, GrowthFactor=0.6):
        self.GrowthFactor = GrowthFactor

    def BuildRandomTree(self, Tree, CurrentLimb, CurrentDepth=0):
        # Force a minimum of 2 branches at depth 0 to avoid sticks
        NumBranches = np.random.randint(2, 4) if CurrentDepth == 0 else np.random.randint(0, 2)
        
        for _ in range(NumBranches):
            self._Branch(Tree, CurrentLimb, CurrentDepth + 1)

    def _Branch(self, Tree, ParentLimb, Depth):
        # Apply a lateral offset via Euler angles (sprawl)
        # This prevents the robot from growing straight up
        Sprawl = np.random.uniform(-0.8, 0.8, size=3)
        Sprawl[2] = 0 # Keep Z-axis vertical for limb orientation
        
        ChildLimb, Conn = Tree.AddLimb(
            ParentLimb=ParentLimb,
            ParentSite=ParentLimb.AddSite(Pos=(0, 0, ParentLimb.Height), Euler=Sprawl),
            Radius=np.random.uniform(0.02, 0.03),
            Height=np.random.uniform(0.1, 0.2),
            Density=500.0 # From Registry
        )
        
        Axis = [1, 0, 0] if np.random.rand() > 0.5 else [0, 1, 0]
        Joint = Tree.AddJointToConnection(Conn, "hinge", Axis, -60, 60)
        Tree.AddActuatorToConnection(Conn, Joint, "position")
        
        if Depth < 3:
            self.BuildRandomTree(Tree, ChildLimb, Depth)