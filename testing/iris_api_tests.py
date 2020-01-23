import os
import sys
sys.path.insert(1, 'api')

import requests
import json as J

# import get_prediction method for accessing api
from get_prediction import get_prediction

# import iris dataset
from sklearn.datasets import load_iris

# import model logic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
iris = load_iris()

# features of the dataset
X = iris.data
# classes of the dataset
y = iris.target

x_test_many, x_test_few, y_test_many, y_test_few = train_test_split(X, y, test_size=0.01, random_state=44)

print('\n\n----- TEST 1: Call API with small dataset -----')
predictions_few = get_prediction(x_test_few)
if predictions_few == False:
    print('TEST 1 FAILED TO RETRIEVE PREDICTIONS')
else:
    print('PREDICTIONS:')
    print(predictions_few, '\n')
    print('CORRECT LABELS:')
    print(y_test_few, '\n')
    print('ACCURACY: ', accuracy_score(y_test_few, predictions_few))

print('\n\n----- TEST 2: Request with large dataset -----')
predictions_many = get_prediction(x_test_many)
if predictions_many == False:
    print('TEST 2 FAILED TO RETRIEVE PREDICTIONS')
else:
    print('PREDICTIONS:')
    print(predictions_many, '\n')
    print('CORRECT LABELS:')
    print(y_test_many, '\n')
    print('ACCURACY: ', accuracy_score(y_test_many, predictions_many))

print('\n\n----- TEST 3: Request with no data -----')
try:
    empty_request = get_prediction([])
except ValueError:
    print('TEST 3 PRODUCED DESIRED BEHAVIOR')

print('\n\n----- TEST 4: Request with invalid data format -----')
try:
    invalid_request = get_prediction([[0, 0]])
except ValueError:
    print('TEST 4 PRODUCED DESIRED BEHAVIOR')

print('\n\n----- TEST 5: Send request with valid raw HTTP POST request -----')
URL = 'http://0.0.0.0:80/' # must have local server running using iris_api.py
headers = {'content-type': 'application/json'}
dictionary = {
    "0": {"sepalLength": 4.8, "sepalWidth": 3.0, "petalLength": 1.4, "petalWidth": 0.1}, # setosa (0)
    "1": {"sepalLength": 6.4, "sepalWidth": 2.9, "petalLength": 4.3, "petalWidth": 1.3}, # versicolor (1)
    "2": {"sepalLength": 6.3, "sepalWidth": 3.3, "petalLength": 6.0, "petalWidth": 2.5} # virginica (2)
}

r = requests.post(url=URL, data=J.dumps(dictionary), headers=headers)
if r.status_code != 201:
    print('TEST 5 FAILED TO PRODUCE PREDICTIONS')
else:
    response = r.json()
    print('Predictions: {}, {}, {} -- Expected: 0, 1, 2'.format(response['0'], response['1'], response['2']))
