# Simulation/SimulatorWorker.py
import multiprocessing as mp
from Simulation.LocomotionEnv import LocomotionEnv

class SimulatorWorker(mp.Process):
    """
    Isolates physics in a subprocess to ensure stability 
    during parallel tournament training.
    """
    def __init__(self, Tree, TaskQueue, ResultQueue):
        super().__init__()
        self.Tree = Tree
        self.TaskQueue = TaskQueue
        self.ResultQueue = ResultQueue

    def run(self):
        try:
            Env = LocomotionEnv(self.Tree)
            # Training logic handled by the Supervisor calling the Env
            self.ResultQueue.put({"Status": "Success", "Tree": self.Tree})
        except Exception as E:
            self.ResultQueue.put({"Status": "Error", "Message": str(E)})