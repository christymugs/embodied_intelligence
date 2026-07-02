# Simulation/LocomotionEnv.py
import gymnasium as gym
import numpy as np
import mujoco
from Simulation.MujocoBuilder import MujocoBuilder

class LocomotionEnv(gym.Env):
    def __init__(self, Tree):
        self.Builder = MujocoBuilder()
        self.Tree = Tree
        self.XMLPath = f"Models/Candidates/Robot_{self.Tree.UID}.xml"
        self.Builder.Save(self.Tree, self.XMLPath)
        
        self.Model = mujoco.MjModel.from_xml_path(self.XMLPath)
        self.Data = mujoco.MjData(self.Model)
        
        self.ActionSpace = gym.spaces.Box(-1, 1, (self.Model.nu,), dtype=np.float32)
        self.ObservationSpace = gym.spaces.Box(-np.inf, np.inf, (self.Model.nq + self.Model.nv,), dtype=np.float32)

    @property
    def action_space(self): return self.ActionSpace
    @property
    def observation_space(self): return self.ObservationSpace

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.Model, self.Data)
        for _ in range(50): mujoco.mj_step(self.Model, self.Data)
        return np.concatenate([self.Data.qpos, self.Data.qvel]).astype(np.float32), {}

    def step(self, action):
        self.Data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.Model, self.Data)
        
        Obs = np.concatenate([self.Data.qpos, self.Data.qvel]).astype(np.float32)
        
        # Reward the robot for moving, but remove the "upright only" requirement
        # Crawling/rolling is now permitted.
        Velocity = self.Data.qvel[0]
        Height = self.Data.qpos[2] if self.Data.qpos.shape[0] > 2 else 0.0
        
        # Reward = Forward Velocity - Energy - Penalty for being 'too high' (prevents sticks)
        Reward = (Velocity * 10.0) - (0.01 * np.sum(np.square(action))) - abs(Height - 0.15)
        
        # Terminate only if the robot is completely buried or stuck
        Terminated = bool(Height < 0.05)
            
        return Obs, Reward, Terminated, False, {"velocity": Velocity}