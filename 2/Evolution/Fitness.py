# Evolution/Fitness.py
import numpy as np

def CountNodes(Limb):
    """
    Recursively traverse the limb tree to count total body segments.
    """
    Count = 1
    for Conn in Limb.Connections:
        Count += CountNodes(Conn.ChildLimb)
    return Count

def CalculateEnergyEfficiency(RewardHistory, ActionHistory):
    """
    Penalizes high control effort relative to movement gain.
    Cost of Transport (CoT) = Energy Expended / Distance Traveled
    """
    if len(ActionHistory) == 0:
        return 0.0
    
    TotalEnergy = np.sum(np.abs(ActionHistory))
    TotalDistance = max(0.001, np.sum(RewardHistory))
    
    return TotalEnergy / TotalDistance

def CalculateLearnability(RewardHistory):
    """
    Baldwin Effect proxy: Measures the slope of the reward curve.
    """
    if len(RewardHistory) < 10:
        return 0.0
    
    Midpoint = len(RewardHistory) // 2
    Gradient = np.gradient(RewardHistory[:Midpoint])
    return np.mean(Gradient)

def GetTotalFitness(Individual, RewardHistory, ActionHistory, ComplexityWeight=0.1):
    """
    Fitness Function for Emergent Morphology:
    Fitness = (Performance) + (Learnability_Bonus) - (Energy_Penalty) - (Complexity_Penalty)
    """
    if not RewardHistory or np.sum(RewardHistory) == 0:
        return -100.0
        
    PeakPerformance = np.max(RewardHistory)
    NumLimbs = CountNodes(Individual.Tree.Root)
    ComplexityPenalty = NumLimbs * ComplexityWeight
    EnergyPenalty = CalculateEnergyEfficiency(RewardHistory, ActionHistory) * 0.05
    LearnabilityBonus = CalculateLearnability(RewardHistory) * 2.0
    
    Fitness = PeakPerformance + LearnabilityBonus - EnergyPenalty - ComplexityPenalty
    
    return float(Fitness)