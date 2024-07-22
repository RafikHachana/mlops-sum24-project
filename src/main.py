import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra
from omegaconf import DictConfig
from src.data import extract_data_training, load_features_training
from src.model import train, evaluate, log_metadata



def run(args):
    cfg = args

    train_data_version = cfg.train_data_version

    X_train, y_train = load_features_training(name = "features_target", version=train_data_version)

    test_data_version = cfg.test_data_version

    X_test, y_test = load_features_training(name = "features_target", version=test_data_version)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    gs = train(X_train, y_train, cfg=cfg)

    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)

@hydra.main(config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:

    run(cfg)
    # # Extract data
    # train_data, val_data, test_data = extract_data_training(cfg)

    # # Train the model
    # model = train(train_data, val_data)

    # # Evaluate the model
    # metrics = evaluate(model, test_data)

    # # Log metadata
    # log_metadata(model, metrics, cfg)

if __name__ == "__main__":
    main()
