'''
This file provides the logic for training and exporting a Logistic Regression Model to the
iris classifier API.

The key requirement when uploading new models to the API is that the encoded sklearn .sav
model is uploaded to the api directory with the filename 'model.sav'
'''
import os

# import iris dataset
from sklearn.datasets import load_iris

# import model logic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# import model storage logic
from sklearn.externals import joblib

# load data
iris = load_iris()

# features of the dataset
X = iris.data
# classes of the dataset
y = iris.target

# split data to training and testing sets with 85% training, 15% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=45)

# train model
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# save model to file
path = os.path.join( os.getcwd(), 'api', 'model.sav')
joblib.dump(model, path)

# load model from file (to ensure it works)
loaded_model = joblib.load(path)

# print accuracy
#Test the model
predictions = loaded_model.predict(x_test)
print(predictions, '\n')  # printing predictions

# Check precision, recall, f1-score
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
