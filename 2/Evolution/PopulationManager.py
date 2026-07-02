# Evolution/PopulationManager.py
import random
import copy
import numpy as np
from MorphoGen.RoboticTree import RobotTree, Limb
from MorphoGen.GraphGrammar import MorphologyGenerator
from Evolution.Mutator import Mutator
# Import from Utils
from Evolution.Utils import GetAllLimbs 

class Individual:
    def __init__(self, Genotype, Tree: RobotTree):
        self.Genotype = Genotype
        self.Tree = Tree
        self.Fitness = 0.0
        self.TrainingHistory = []

class PopulationManager:
    def __init__(self, Size=20):
        self.Size = Size
        self.Population = [self._CreateRandomIndividual() for _ in range(Size)]

    def _CreateRandomIndividual(self):
        Root = Limb(None, "Limb0", 0.05, 0.5, 1000, LimbID=0)
        Tree = RobotTree(Root)
        Generator = MorphologyGenerator(GrowthFactor=np.random.uniform(0.3, 0.8))
        Generator.BuildRandomTree(Tree, Root)
        return Individual(Genotype={'Growth': 0.6}, Tree=Tree)

    def Evolve(self):
        self.Population.sort(key=lambda x: x.Fitness, reverse=True)
        Survivors = self.Population[:self.Size // 2]
        NewPopulation = []
        for _ in range(self.Size):
            Parent = random.choice(Survivors)
            ChildTree = copy.deepcopy(Parent.Tree)
            Child = Individual(Genotype=Parent.Genotype.copy(), Tree=ChildTree)
            Mutator.Mutate(Child)
            NewPopulation.append(Child)
        self.Population = NewPopulation