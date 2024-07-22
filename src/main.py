import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra
from omegaconf import DictConfig
from src.data import extract_data_training
from src.model import train, evaluate, log_metadata

@hydra.main(config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    # Extract data
    train_data, val_data, test_data = extract_data_training(cfg)

    # Train the model
    model = train(train_data, val_data)

    # Evaluate the model
    metrics = evaluate(model, test_data)

    # Log metadata
    log_metadata(model, metrics, cfg)

if __name__ == "__main__":
    main()
