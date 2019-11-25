"""A simple example flask application
"""
from flask import Flask, jsonify, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
import joblib
import cv2
import traceback

clear_session()
app = Flask(__name__)
canine_model = None

classes = """n02085620-Chihuahua			
n02100583-vizsla
n02085782 - Japanese_spaniel
n02100735 - English_setter
n02085936 - Maltese_dog
n02100877 - Irish_setter
n02086079 - Pekinese
n02101006 - Gordon_setter
n02086240 - Shih - Tzu
n02101388 - Brittany_spaniel
n02086646 - Blenheim_spaniel
n02101556 - clumber
n02086910 - papillon
n02102040 - English_springer
n02087046 - toy_terrier
n02102177 - Welsh_springer_spaniel
n02087394 - Rhodesian_ridgeback
n02102318 - cocker_spaniel
n02088094 - Afghan_hound
n02102480 - Sussex_spaniel
n02088238 - basset
n02102973 - Irish_water_spaniel
n02088364 - beagle
n02104029 - kuvasz
n02088466 - bloodhound
n02104365 - schipperke
n02088632 - bluetick
n02105056 - groenendael
n02089078 - black - and -tan_coonhound
n02105162 - malinois
n02089867 - Walker_hound
n02105251 - briard
n02089973 - English_foxhound
n02105412 - kelpie
n02090379 - redbone
n02105505 - komondor
n02090622 - borzoi
n02105641 - Old_English_sheepdog
n02090721 - Irish_wolfhound
n02105855 - Shetland_sheepdog
n02091032 - Italian_greyhound
n02106030 - collie
n02091134 - whippet
n02106166 - Border_collie
n02091244 - Ibizan_hound
n02106382 - Bouvier_des_Flandres
n02091467 - Norwegian_elkhound
n02106550 - Rottweiler
n02091635 - otterhound
n02106662 - German_shepherd
n02091831 - Saluki
n02107142 - Doberman
n02092002 - Scottish_deerhound
n02107312 - miniature_pinscher
n02092339 - Weimaraner
n02107574 - Greater_Swiss_Mountain_dog
n02093256 - Staffordshire_bullterrier
n02107683 - Bernese_mountain_dog
n02093428 - American_Staffordshire_terrier
n02107908 - Appenzeller
n02093647 - Bedlington_terrier
n02108000 - EntleBucher
n02093754 - Border_terrier
n02108089 - boxer
n02093859 - Kerry_blue_terrier
n02108422 - bull_mastiff
n02093991 - Irish_terrier
n02108551 - Tibetan_mastiff
n02094114 - Norfolk_terrier
n02108915 - French_bulldog
n02094258 - Norwich_terrier
n02109047 - Great_Dane
n02094433 - Yorkshire_terrier
n02109525 - Saint_Bernard
n02095314 - wire - haired_fox_terrier
n02109961 - Eskimo_dog
n02095570 - Lakeland_terrier
n02110063 - malamute
n02095889 - Sealyham_terrier
n02110185 - Siberian_husky
n02096051 - Airedale
n02110627 - affenpinscher
n02096177 - cairn
n02110806 - basenji
n02096294 - Australian_terrier
n02110958 - pug
n02096437 - Dandie_Dinmont
n02111129 - Leonberg
n02096585 - Boston_bull
n02111277 - Newfoundland
n02097047 - miniature_schnauzer
n02111500 - Great_Pyrenees
n02097130 - giant_schnauzer
n02111889 - Samoyed
n02097209 - standard_schnauzer
n02112018 - Pomeranian
n02097298 - Scotch_terrier
n02112137 - chow
n02097474 - Tibetan_terrier
n02112350 - keeshond
n02097658 - silky_terrier
n02112706 - Brabancon_griffon
n02098105 - soft - coated_wheaten_terrier
n02113023 - Pembroke
n02098286 - West_Highland_white_terrier
n02113186 - Cardigan
n02098413 - Lhasa
n02113624 - toy_poodle
n02099267 - flat - coated_retriever
n02113712 - miniature_poodle
n02099429 - curly - coated_retriever
n02113799 - standard_poodle
n02099601 - golden_retriever
n02113978 - Mexican_hairless
n02099712 - Labrador_retriever
n02115641 - dingo
n02099849 - Chesapeake_Bay_retriever
n02115913 - dhole
n02100236 - German_short - haired_pointer
n02116738 - African_hunting_dog
""".split('\n')


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
    global classes
    try:
        image = request.files['file'].read()
        # https://stackoverflow.com/a/27537664/818687
        # arr = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)
        arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        print("arr.shape", arr.shape)
        my_image = arr / 255.0
        print("my_image", my_image.shape)
        my_image = cv2.resize(my_image, (256, 256))
        print("my_image1", my_image.shape)
        my_image = np.expand_dims(my_image, axis=0)
        print("my_image2", my_image.shape)
        print("Got here")

        pred = canine_model.predict(my_image)[0]
        print("pred", pred)
        prediction = pred.argmax(axis=-1)
        actual_prediction = pred[prediction]
        print("prediction", prediction)
        sorted_probabilities = sorted(pred[0], reverse=True)

        # for prob in sorted_probabilities:
        #     index = np.where(pred[0] == prob)[0][0]
        #     print("%s - %.2f%% probability" % (classes[index], pred[0][index] * 100))
        #     if prob < .01:
        #         break
        return jsonify({
            "message": "%s - %.2f%% probability" % (classes[prediction], pred[0][0] * 100)
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "message": f"An error occurred. {e}"
        })


# @app.route("/canine-ui")
# def mnist_ui():
#     return render_template("Website.html")


@app.route("/canine-ui")
def mnist_ui():
    return render_template("mnist.html")


if __name__ == '__main__':
    load_mnist_model()
    app.run(debug=True)
