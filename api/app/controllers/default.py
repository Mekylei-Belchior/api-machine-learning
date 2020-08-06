""" Imported Modules """
from pickle import load
from pandas import DataFrame
from app import app

from flask import request


# Load pickle file model
with open('C:/Users/mekyl/Documents/GitHub/api-machine-learning/model/model.pkl', 'rb') as file:
    model = load(file)


@app.route('/api_predict', methods=['POST'])
@app.route('/', methods=['POST'])
def api_predict():
    """ Route to API Predict """
    validation_json = request.get_json()

    if validation_json:
        if isinstance(validation_json, dict):
            data = DataFrame(validation_json, index=[0])
        else:
            data = DataFrame(validation_json, columns=validation_json[0].keys())

	# Prediction
    pred = model.predict(data)

    data['prediction'] = pred

    return data.to_json(orient='records')
