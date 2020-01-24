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
The API can be accessed in two ways (from a seperate terminal):
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

I think there are a number of ways to handle this fundamental Big Data problem, in the system I
built for this particular problem I did all of the training on my computer, then sent the trained
model to the server (really just a different local directory given I'm also hosting the server locally)
to serve predictions, so to handle the scaling problem I'd have to implement a pretty simple
batching algorithm where I'm pulling small chunks of data from the database for training and sort of
iterating through the desired amount of batches.

As for a more expensive option you could distribute the training load amongst one or more
GPUs, increasing both training speed and RAM space.

2. Describe your optimal versioning strategy for APIs which expose machine learning
models. How does training the model on new data fit into versioning strategy? List the
pros and cons of your described strategy in detail.

I think the key to a successful versioning strategy is consistency amongst existing capabilities.
What I mean by that is the methods and structures that your users have been working with should
remain functional as new features become available. For example, if I wanted to improve the 
functionality of the get_prediction method to where the user can specify which pre-trained sklearn
model they want to use, I would instead just write a whole new method, say get_prediction_with_model(model, data).
Of course this approach could end up in some level of redundancy and duplication, but as a programmer I'd
rather not have my existing code broken by new features that I may or may not be interested in.

As for training new models on more data I would just make sure that again previous versions of models
remain available, so users have proper time to adjust. In summary:

Pros: API remains backwards compatible as new versions are released, allowing users to choose whether
to implement new capabilities.
Cons: Source code could become crowded and redundant and API becomes resistant to fundamental/large-scale
improvements.

3. Describe your choice of model and how it fits the problem. List benefits and drawbacks
of this type of model used in the way you have chosen and where there may be scaling
issues as a system like this grows in size or complexity.

I chose to use logistic regression in sklearn to fit this particular problem. I chose logistic regression
because it's an easy-to-apply algorithm and the iris dataset is famously easy to make a classifier for. I
wanted to make sure I wasn't spending hours on a super complex algorithm when I could be focusing on the API development, something that I had to learn some new frameworks for and I knew might take a while.

The pros of how I went about deploying the logistic regression model is that I used sklearn's joblib to encode the
model to a file for later use, which makes transitioning to other sklearn models and using them in the API
quite simple. Additionally logistic regression takes relatively few iterations to converge which is always nice.

The obvious con to this approach though is the API is limited exclusively to sklearn
models, which definitely isn't the most popular library if you want to move toward more advanced machine
learning algorithms. If I were to push this API to its full potential it would certainly include the
ability to use different algorithms from different libraries.