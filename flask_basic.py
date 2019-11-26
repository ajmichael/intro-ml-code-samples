"""A simple example flask application
"""
from flask import Flask, jsonify, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
import joblib
import cv2
import traceback
from dogs import dogs

clear_session()
app = Flask(__name__)
canine_model = None


def load_mnist_model():
    global canine_model
    model_file = 'models/model.hdf5'
    canine_model = load_model(model_file)
    canine_model._make_predict_function()


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/variables/<variable>")
def example_variable(variable):
    return jsonify({
        "message": f"The variable you entered is {variable}"
    })


@app.route("/request-args")
def example_request_args():
    try:
        a = request.args["a"]
        b = request.args["b"]
        c = request.args["c"]
        return jsonify({
            "message": f"You entered a = {a}, b= {b} and c= {c}."
        })
    except:
        return jsonify({
            "message": f"You did not provide one of a, b, or c."
        })


@app.route("/canine", methods=["POST"])
def mnist_predict():
    try:
        # gets file as bytes
        image = request.files['file'].read()
        # https://stackoverflow.com/a/27537664/818687
        # returns image
        arr = cv2.imdecode(np.frombuffer(
            image, np.uint8), cv2.IMREAD_UNCHANGED)
        # makes it (1, 255, 255, 3)
        my_image = np.expand_dims(cv2.resize(arr / 255.0, (256, 256)), axis=0)

        # predict, returns list of a list of probabilities for each class
        # it is just a list
        probabilities = canine_model.predict(my_image)[0]
        # gets best location
        location_of_best = probabilities.argmax(axis=-1)
        # the probability of best
        best_probability = probabilities[location_of_best]

        return jsonify({
            "message": "%s - %.2f%% probability" % (dogs[location_of_best], best_probability * 100)
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "message": f"An error occurred. {e}"
        })


@app.route("/canine-ui")
def canine_ui():
    return render_template("Website.html")


@app.route("/canine-ui-old")
def mnist_ui():
    return render_template("mnist.html")


if __name__ == '__main__':
    load_mnist_model()
    app.run(debug=True)
