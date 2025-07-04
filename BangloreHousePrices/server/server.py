from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route("/hello")
def hello():
    return "Hi, Abhijeet!"


@app.route("/get_location_names")
def get_location_names():
    response = jsonify({"locations": util.get_location_names()})
    response.headers.add("Access-control-Allow-Origin", "*")

    return response


if __name__ == "__main__":
    print("Starting Python Flask File Server For Home Price Prediction...!")
    app.run()
