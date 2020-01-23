import requests
import json as J

def get_prediction(data):
    # do some basic validation
    if len(data) < 1:
        raise ValueError('Data array was empty.')
    if len(data[0]) != 4:
        raise ValueError('Data has incorrect number of features, input should not include class label')
    
    # populate a dictionary with data to later be sent as json to API
    dictionary = {}
    for i in range(len(data)):
        d = data[i]
        dictionary[str(i)] = {"sepalLength": d[0], "sepalWidth": d[1], "petalLength": d[2], "petalWidth": d[3]}

    # send a post request to the API
    URL = 'http://0.0.0.0:80/' # must have local server running using iris_api.py
    headers = {'content-type': 'application/json'}
    
    r = requests.post(url=URL, data=J.dumps(dictionary), headers=headers)  # call the API
    
    # check for valid response code
    if r.status_code != 201:
        return False

    r = r.json()  # parse json from response

    predictions = []
    for i in range(len(data)): # return column of predictions as python list
        predictions.append(r[f'{i}'])
    return predictions