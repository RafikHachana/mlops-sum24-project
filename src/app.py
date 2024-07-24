import gradio as gr
# import mlflow
# from utils import init_hydra
# from model import load_features
# from transform_data import transform_data
import json
import requests
import numpy as np
import pandas as pd
import yaml
# cfg = init_hydra()

port_number = 5001

model_sig = requests.get(f"http://localhost:{port_number}/info").text

inputs_list = json.loads(yaml.safe_load(model_sig)['signature']["inputs"])
# You need to define a parameter for each column in your raw dataset
def predict(*args):
    
    # This will be a dict of column values for input data sample
    features = dict(zip([x['name'] for x in inputs_list], args))
    
    # print(features)
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    # X = None 
    # transform_data(
    #                     df = raw_df, 
    #                     cfg = cfg, 
    #                     return_df = False, 
    #                     only_transform = True, 
    #                     transformer_version = "v4", 
    #                     only_X = True
    #                   )
    
    # Convert it into JSON
    # example = X.iloc[0,:]

    example = json.dumps( 
        { "inputs": features }
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:5001/predict",
        json={ "inputs": features },
        headers={"Content-Type": "application/json"},
    )
    
    print(response.json())
    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    return response.json()['prediction']

# Mapping MLflow types to Gradio components
def mlflow_to_gradio(input_type, label):
    if label in ['Windows', 'Linux', 'Mac']:
        return gr.Checkbox(label=label)
    if input_type in ["long", "integer", "int"]:
        return gr.Number(label=label)
    elif input_type == "double":
        return gr.Number(label=label, precision=2)
    elif input_type == "float":
        return gr.Number(label=label)
    elif input_type == "string":
        return gr.Textbox(label=label)
    else:
        return gr.Textbox(label=label)

# Only one interface is enough
demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict,
    
    # Here, the arguments in `predict` function
    # will populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        mlflow_to_gradio(input_type=x['type'], label=x['name']) for x in inputs_list
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="Prediction result"),
    
    # This will provide the user with examples to test the API
    # examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

# Launch the web UI locally on port 5155
demo.launch(server_port = 5155, share=True)
