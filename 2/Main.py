# Main.py
from Supervisor import Supervisor
import argparse

def main():
    parser = argparse.ArgumentParser(description="Embodied Intelligence Research Pipeline")
    parser.add_argument("--gen", type=int, default=20, help="Number of generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    args = parser.parse_args()

    print("🚀 Initializing Embodied Intelligence Pipeline...")
    
    # Initialize the Orchestrator
    ResearchSupervisor = Supervisor(PopulationSize=args.pop)
    
    # Run the evolutionary loop
    try:
        ResearchSupervisor.RunPipeline(Generations=args.gen)
        print("\n✅ Research Run Complete. Data saved to Experiments/Logs/")
    except KeyboardInterrupt:
        print("\n⏸️ Pipeline paused by user. Saving state...")

if __name__ == "__main__":
    main()