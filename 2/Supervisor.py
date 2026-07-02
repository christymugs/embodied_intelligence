# Supervisor.py
import os
from Evolution.PopulationManager import PopulationManager
from Evolution.Fitness import GetTotalFitness
from Simulation.LocomotionEnv import LocomotionEnv
from Experiments.ExperimentLogger import ExperimentLogger
from stable_baselines3 import PPO

class Supervisor:
    def __init__(self, PopulationSize=10):
        self.PopManager = PopulationManager(Size=PopulationSize)
        self.Logger = ExperimentLogger()
        self.ResultsDir = "Models/Candidates/"
        os.makedirs(self.ResultsDir, exist_ok=True)

    def RunPipeline(self, Generations=20):
        for Gen in range(Generations):
            print(f"\n--- Generation {Gen}: Phase 1 (The Sieve) ---")
            
            for Ind in self.PopManager.Population:
                Env = LocomotionEnv(Ind.Tree)
                Model = PPO("MlpPolicy", Env, verbose=0)
                
                # Train and capture rewards
                Model.learn(total_timesteps=10000)
                
                # Simple evaluation to capture reward history for fitness
                Obs, _ = Env.reset()
                RewardHistory = []
                for _ in range(100):
                    Action, _ = Model.predict(Obs)
                    Obs, Rew, Done, _, _ = Env.step(Action)
                    RewardHistory.append(Rew)
                    if Done: break
                
                Ind.Fitness = GetTotalFitness(Ind, RewardHistory, [])
                Ind.TrainingHistory = RewardHistory

            self.PopManager.Population.sort(key=lambda x: x.Fitness, reverse=True)
            TopPerformers = self.PopManager.Population[:self.PopManager.Size // 2]
            
            print(f"--- Generation {Gen}: Phase 3 (Champion Training) ---")
            for Ind in TopPerformers:
                Env = LocomotionEnv(Ind.Tree)
                Model = PPO("MlpPolicy", Env, verbose=0)
                Model.learn(total_timesteps=100000)
                
            self.PopManager.Evolve()
            self.Logger.LogGeneration(Gen, self.PopManager.Population)
            print(f"Gen {Gen} Best Fitness: {max(Ind.Fitness for Ind in self.PopManager.Population):.2f}")