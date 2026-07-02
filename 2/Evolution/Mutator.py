# Evolution/Mutator.py
import random
import numpy as np
from Evolution.Utils import GetAllLimbs

class Mutator:
    @staticmethod
    def Mutate(Individual):
        MutationType = random.choice(['Add', 'Scale'])
        
        if MutationType == 'Add':
            TargetLimb = random.choice(GetAllLimbs(Individual.Tree.Root))
            NewLimb, Conn = Individual.Tree.AddLimb(
                ParentLimb=TargetLimb,
                # FIX: Change pos to Pos here as well
                ParentSite=TargetLimb.AddSite(Pos=(0, 0, TargetLimb.Height)),
                Radius=random.uniform(0.02, 0.04),
                Height=random.uniform(0.1, 0.2),
                Density=800
            )
            Joint = Individual.Tree.AddJointToConnection(Conn, "hinge", [0, 1, 0], -45, 45)
            Individual.Tree.AddActuatorToConnection(Conn, Joint, "position")

        elif MutationType == 'Scale':
            Target = random.choice(GetAllLimbs(Individual.Tree.Root))
            Target.Height = np.clip(Target.Height * random.uniform(0.9, 1.1), 0.05, 0.5)
            Target.Radius = np.clip(Target.Radius * random.uniform(0.9, 1.1), 0.02, 0.06)