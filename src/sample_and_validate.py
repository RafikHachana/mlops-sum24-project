import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import sample_data, validate_initial_data, run_checkpoint


if __name__ == '__main__':
    # Sample data
    sample_data()

    # Validate initial data
    # validate_initial_data()

    # Run checkpoint
    # if not run_checkpoint():
    #     raise Exception("Checkpoint failed!")
