from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    a = float(request.form['Sepal Length'])
    b = float(request.form['Sepal width'])
    c = float(request.form['Petal Length'])
    d = float(request.form['Petal width'])

    model = pickle.load(open("model.pkl","rb"))
    prediction = model.predict([[a,b,c,d]])

    last = prediction[0]
    
    message="NOTE : This model predicts the species within an accuracy of 96.666667"
    return render_template("index.html", prediction_text="The predicted species is {}".format(last), message=message)

if __name__=="__main__":
    app.run()