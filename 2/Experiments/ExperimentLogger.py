# Experiments/ExperimentLogger.py
import csv
import os
import numpy as np

class ExperimentLogger:
    def __init__(self, LogDir="Experiments/Logs/"):
        self.LogDir = LogDir
        os.makedirs(self.LogDir, exist_ok=True)
        self.GenFile = os.path.join(self.LogDir, "GenerationLog.csv")
        self.IndFile = os.path.join(self.LogDir, "IndividualLog.csv")
        
        with open(self.GenFile, 'w', newline='') as f:
            csv.writer(f).writerow([
                "Generation", "MeanFitness", "StdFitness", "MeanComplexity", 
                "BestFitness", "BestComplexity", "MeanEnergyEfficiency"
            ])
            
        with open(self.IndFile, 'w', newline='') as f:
            csv.writer(f).writerow(["Generation", "IndID", "Fitness", "Complexity", "EnergyEff"])

    def LogGeneration(self, Gen, Population):
        from Evolution.Fitness import CountNodes, CalculateEnergyEfficiency
        
        Fitnesses = [Ind.Fitness for Ind in Population]
        Complexities = [CountNodes(Ind.Tree.Root) for Ind in Population]
        # Assuming training histories are stored in Ind.TrainingHistory
        Efficiencies = [CalculateEnergyEfficiency(Ind.TrainingHistory, []) for Ind in Population]
        
        BestIdx = np.argmax(Fitnesses)
        
        # Log Summary Statistics
        with open(self.GenFile, 'a', newline='') as f:
            csv.writer(f).writerow([
                Gen, np.mean(Fitnesses), np.std(Fitnesses), np.mean(Complexities),
                np.max(Fitnesses), Complexities[BestIdx], np.mean(Efficiencies)
            ])
            
        # Log Top 3 Individuals' Details for "Hall of Fame" tracking
        SortedIndices = np.argsort(Fitnesses)[-3:]
        with open(self.IndFile, 'a', newline='') as f:
            for Idx in SortedIndices:
                csv.writer(f).writerow([
                    Gen, Population[Idx].Tree.UID, Fitnesses[Idx], 
                    Complexities[Idx], Efficiencies[Idx]
                ])