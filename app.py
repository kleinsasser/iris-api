from flask import Flask, request
from flask_restful import Resource, Api

# import iris dataset
from sklearn.datasets import load_iris

# import model logic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# import model storage logic
from sklearn.externals import joblib

iris = load_iris()
# features of the dataset
X = iris.data
# classes of the dataset
y = iris.target
# split data to training and testing sets with 80% training, 20% testing
_, x_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=44)

app = Flask(__name__)
api = Api(app)

class IrisClassifier(Resource):
    def get(self):                
        filename = 'logistic_regression_model.sav'
        loaded_model = joblib.load(filename)

        # print accuracy
        #Test the model
        predictions = loaded_model.predict(x_test)
        #Check precision, recall, f1-score
        accuracy = accuracy_score(y_test, predictions)

        return {'predictions': '{}'.format(predictions), 'accuracy': '{}'.format(accuracy)}, 200
    
    def post(self):
        json = request.get_json()
        return {'echo': json}, 201

api.add_resource(IrisClassifier, '/')

if __name__ == '__main__':
    app.run(debug = True)