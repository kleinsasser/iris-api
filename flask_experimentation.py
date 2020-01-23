from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class IrisClassifier(Resource):
    def post(self):
        json = request.get_json()
        return {'echo': json}, 201

api.add_resource(IrisClassifier, '/')

if __name__ == '__main__':
    app.run(debug = True)