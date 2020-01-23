# import flask and json
from flask import Flask, request
from flask_restful import Resource, Api
import json as J

# import necessary sklearn functions
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# set up flask app/api object
app = Flask(__name__)
api = Api(app)

# add post handling functionality
class IrisClassifier(Resource):
    def post(self):
        data = request.get_json() # get data from post

        x = [] # list for storing data
        for i in range(len(data)):
            d = [] # row to be appended to x
            d.append(data[f'{i}']['sepalLength'])
            d.append(data[f'{i}']['sepalWidth'])
            d.append(data[f'{i}']['petalLength'])
            d.append(data[f'{i}']['petalWidth'])
            x.append(d)

        # load model from model.sav file and make prediction
        filename = "model.sav" # this file should exist in the same directory as this file
        model = joblib.load(filename)
        predictions = model.predict(x)

        response = {} # format prediction in dictionary
        for i in range(len(predictions)):
            response[str(i)] = int(predictions[i])
        
        return response, 201

api.add_resource(IrisClassifier, '/') # add iris classifier to api at root URL

if __name__ == '__main__':
    # run api on specific local URL
    app.run(debug = True, host='0.0.0.0', port='80')