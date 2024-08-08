from flask import Flask,request,render_template
import  pickle
from keras.models import load_model
import numpy as np

# loading models
app = Flask(__name__)
models = load_model("model.ann.h5")
scaler = pickle.load(open('scalar.pkl','rb'))

# creating routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/house",methods=['POST','GET'])
def house():
    if request.method=='POST':
       longitude = request.form['longitude']
       latitude = request.form['latitude']
       houseage = request.form['houseage']
       houserooms = request.form['houserooms']
       totlabedrooms = request.form['totlabedrooms']
       population = request.form['population']
       households = request.form['households']
       medianincome = request.form['medianincome']
       oceanproximity = request.form['oceanproximity']

       features = np.array([longitude,latitude,houseage,houserooms,totlabedrooms,population,households,
                            medianincome,oceanproximity], dtype=float)

       features_scaled = scaler.transform([features])

       price = models.predict(features_scaled).reshape(1,-1)
       return render_template('index.html',result = price)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/doc")
def doc():
    return render_template('doc.html')

if __name__ == "__main__":
    app.run(debug=True,port=5003)