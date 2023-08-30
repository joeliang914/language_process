import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle, bz2
import _pickle as cPickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = bz2.BZ2File('./models/model.pkl.bz2','rb')
prediction_model = cPickle.load(model)
count_vect = pickle.load(open('./models/count_vect.pkl', 'rb'))
label_encoder = pickle.load(open('./models/label_encoder.pkl', 'rb'))

@app.route('/', methods = ["GET", "POST"])
def home():

    return render_template('index.html')
    

@app.route("/predict", methods=["POST"])
def predict():
    #count_vect = CountVectorizer()
    #label_encoder = LabelEncoder()
    if request.method == "POST":
        text = request.form.get('language')

    x = count_vect.transform([text]).toarray()
    prediction = prediction_model.predict(x)
    prediction = label_encoder.inverse_transform(prediction)
    #x = count_vect.transform([text]).toarray()
    #x = count_vect.transform(text).toarray() # converting text to bag of words model (Vector)
    #lang = model.predict(x) # predicting the language
    #lang = label_encoder.inverse_transform(lang) # finding the language corresponding the the predicted value
    return render_template('index.html', prediction = prediction[0], text = text)
    
if __name__ == "__main__":
    app.run(debug=True)
