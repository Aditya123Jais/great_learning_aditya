# import necessary libraries
import pandas as pd
from flask import Flask, request, Response
from joblib import load

# Loading my model
my_logit_model = load('Model/my_logistic_model.joblib')

# Initialising
app = Flask(__name__)


@app.route("/get_logistic_predictions", methods = ['POST','GET'])
def get_logistic_predictions():
    data = request.json     # Reading user input data
    column_1 = data.get('SepalLengthCm')
    column_2 = data.get('SepalWidthCm')
    column_3 = data.get('PetalLengthCm')
    column_4 = data.get('PetalWidthCm')

    # You want to use the users data and give it to your model
    model_prediction = my_logit_model.predict([[column_1, column_2, column_3, column_4]])

    # Getting just the number from the array
    my_prediction = model_prediction[0]

    # Giving the user the answer we are looking for
    return Response(str(my_prediction))


if __name__ == '__main__':
    app.run(debug=True)