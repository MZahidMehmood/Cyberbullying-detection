import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from baselines import train_evaluate_baselines

if __name__ == "__main__":
    train_evaluate_baselines()
