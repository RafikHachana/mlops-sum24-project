from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os
import requests
import json
import pandas as pd
import yaml

BASE_PATH = os.path.expandvars("$PROJECTPATH")
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():

	response = make_response(str(model.metadata), 200)
	response.content_type = "text/plain"
	return response

@app.route("/", methods = ["GET"])
def home():
	msg = """
	Welcome to our ML service to predict Customer satisfaction\n\n

	This API has two main endpoints:\n
	1. /info: to get info about the deployed model.\n
	2. /predict: to send predict requests to our deployed model.\n

	"""

	response = make_response(msg, 200)
	response.content_type = "text/plain"
	return response

# /predict endpoint
@app.route("/predict", methods = ["POST"])
def predict():
    # Get schema

    schema = {x['name']: x['type'] for x in json.loads(yaml.safe_load(str(model.metadata))['signature']["inputs"])}
	
    # EDIT THIS ENDPOINT
    print(json.loads(request.data.decode("utf-8")))
    payload = json.loads(request.data.decode("utf-8"))

    inputs = pd.DataFrame({k: [int(v) if schema[k] in ["int", "long"] else float(v)] for k,v in payload['inputs'].items()})

    response = model.predict(inputs)

    # EXAMPLE
    # content = str(request.data)
    # response = make_response(content, 200)
    # response.headers["content-type"] = "application/json"
    print(response)
    return jsonify({"prediction":response[0]})

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)