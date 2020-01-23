# Iris Dataset Classifier API
This project was written in Python 3.8.1 but should be compatible with any Python 3 version.
This project depends on the following python packages, make sure you have them installed in your Python interpreter.
- sklearn
- joblib
- json
- flask
- flask_restful
- requests

## How to run the API
### 1. Set up a local server running the API
a. Clone the iris-api repository to your computer
b. Navigate to the repo in a terminal
c. Run the api/iris_api.py file in a terminal (python3 api/iris_api.py)

### 2. Call the API
The API can be accessed in two ways:
#### 1. Using an HTTP POST request to http://0.0.0.0/80 (painful)
The request should contain a JSON object in the following format, ensuring your mime-type is 'application/json'
<pre><code>
{
    "0": {"sepalLength": 5.0, "sepalWidth": 3.6, "petalLength": 1.4, "petalWidth": 0.2},
    "1": {"sepalLength": 4.2, "sepalWidth": 3.9, "petalLength": 1.0, "petalWidth": 0.4},
    "2": {"sepalLength": 6.0, "sepalWidth": 2.9, "petalLength": 1.6, "petalWidth": 0.2,}
    ...
}
</code></pre>

The response will include a JSON file in the following format with the model's prediction:
<pre><code>
{
    "0": 0
    "1": 2
    "2": 1
    ...
}
</code></pre>

#### 2. Using the get_prediction python method from a script (easy)
Simply import the method using (you will likely have to add the 'api' directory to your sys.path, see iris_api_tests.py)
<pre><code>
from get_prediction import get_prediction
</code></pre>
The method takes one argument, a python list containing the data for prediction:
<pre><code>
[
    [5.0, 3.6, 1.4, 0.2],
    ...
]
</code></pre>
and returns a python list of the model predictions from the API:
<pre><code>
[
    0,
    ...
]
</code></pre>

Run the iris_api_tests.py script to see both behaviors in action.

## Follow-up Questions
1. This dataset was obviously quite small, in the product you will be working with much
more data. How would you scale your training pipeline and/or model to handle datasets
which do not easily fit into system memory?



2. Describe your optimal versioning strategy for APIs which expose machine learning
models. How does training the model on new data fit into versioning strategy? List the
pros and cons of your described strategy in detail.



3. Describe your choice of model and how it fits the problem. List benefits and drawbacks
of this type of model used in the way you have chosen and where there may be scaling
issues as a system like this grows in size or complexity.

