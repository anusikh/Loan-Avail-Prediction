from flask import Flask, render_template, request
import pickle
import numpy as np

#Load the Random Forest Classifier model
filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


#'Credit_History','Education','Married','Property_Area','CoapplicantIncome'
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        cred = int(request.form['Credit_History'])
        ed = int(request.form['Education'])
        mar = int(request.form['Married'])
        prop_area = int(request.form['Property_Area'])
        coapp = float(request.form['CoapplicantIncome'])


        data = np.array([[cred, ed, mar, prop_area, coapp]])
        my_pred = classifier.predict(data)

        return render_template('result.html', prediction=my_pred)
        
        

if __name__ == '__main__':
    app.run()
