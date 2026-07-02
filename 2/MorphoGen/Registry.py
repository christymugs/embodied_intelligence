# MorphoGen/Registry.py

class RobotRegistry:
    """
    Centralized constraints. High torso density anchors the robot.
    """
    MinRadius = 0.02
    MaxRadius = 0.06
    MinHeight = 0.1
    MaxHeight = 0.3
    
    # Force the base to be heavy (Torso) and appendages light (Limb)
    # This prevents the evolution of "floating" or top-heavy pogo sticks
    TorsoDensity = 1500.0
    LimbDensity = 500.0
    
    @staticmethod
    def GetDensity(IsRoot):
        return RobotRegistry.TorsoDensity if IsRoot else RobotRegistry.LimbDensity