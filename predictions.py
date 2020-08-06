""" Modules import """

from json import loads
from requests import post
from pandas import DataFrame

from sklearn.datasets import load_iris

# Dataset loading
iris = load_iris()

features = iris['data']
labels = iris['feature_names']
target = iris['target']

target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

df = DataFrame(features, columns=labels)

sample = df.sample()
sample_json = sample.to_json(orient='records')


# API parameters
URL = 'http://127.0.0.1:5000/api_predict'
data = sample_json
header = {'Content-type': 'application/json'}

# API response
r = post(url=URL, data=data, headers=header)

data_dict = loads(data)[0]
response = r.json()[0]

# Result

print()

# Print predict data
print('#' * 20, f'{"Predict Data":^20}', '#' * 20)
for k, v in data_dict.items():
    print(f'{k}: {v}\n', end='')

print('\n\n')

# Print predict response
print('#' * 20, f'{"Predict Response":^20}', '#' * 20)
for k, v in response.items():
    print(f'{k}: {v}\n', end='')

print(f'\nPredicted Class ({response["prediction"]}): {target_names[response["prediction"]]}')
