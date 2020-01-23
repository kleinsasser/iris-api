import os
import sys
sys.path.insert(1, 'api')

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

# split data to training and testing sets with 80% training, 20% testing
x_test_many, x_test_few, y_test_many, y_test_few = train_test_split(X, y, test_size=0.01, random_state=44)

predictions_many = get_prediction(x_test_many)
print(accuracy_score(y_test_many, predictions_many))