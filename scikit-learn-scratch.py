# import iris dataset
from sklearn.datasets import load_iris

# import model logic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# load data
iris = load_iris()

# features of the dataset
X = iris.data
# classes of the dataset
y = iris.target

# split data to training and testing sets with 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=44)

# train model
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# print accuracy
#Test the model
predictions = model.predict(x_test)
print(predictions, '\n')  # printing predictions

#Check precision, recall, f1-score
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
