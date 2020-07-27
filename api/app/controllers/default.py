from pandas import DataFrame

from flask import request
from pickle import load
from app import app


# Load pickle file model
with open('https://github.com/Mekylei-Belchior/API-MachineLearning/blob/master/model/model.pkl', 'rb') as file:
	model = load(file)


@app.route('/api_predict', methods=['POST'])
@app.route('/', methods=['POST'])
def api_predict():
	validation_json = request.get_json()

	if validation_json:
		if isinstance(validation_json, dict):
			df = DataFrame(validation_json, index=[0])
		else:
			df = DataFrame(validation_json, columns=validation_json[0].keys())

	# Prediction
	pred = model.predict(df)

	df['Prediction'] = pred

	return df.to_json( orient='records')

