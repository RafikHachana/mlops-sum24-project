import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import requests
import hydra
from src.data import load_features_training

@hydra.main(config_path="../configs", config_name="main", version_base=None) # type: ignore
def predict(cfg = None):

    X, y = load_features_training(name ="features_target", 
                        version = "v4", 
                        # random_state=cfg.random_state
                        )

    example = X.iloc[0,:]
    example_target = y[0]

    example = json.dumps( 
    { "inputs": example.to_dict() }
    )

    payload = example

    response = requests.post(
        url=f"http://localhost:{cfg.port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(response.json())
    print("encoded target labels: ", example_target)
    # print("target labels: ", list(cfg.data.labels)[example_target])


if __name__=="__main__":
    predict()
