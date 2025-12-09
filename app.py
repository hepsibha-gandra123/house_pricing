import pickle 
from flask import Flask,app, request, jsonify,url_for,render_template
import numpy as np
import pandas as pd
from param import output

app=Flask(__name__)
regmodel=pickle.load(open('house_pricing/regmodel.pkl','rb'))
scalar=pickle.load(open('house_pricing/scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    final_data=np.array(list(data.values())).reshape(1,-1)
    print(final_data)
    new_data=scalar.transform(final_data)
    output=regmodel.predict(new_data)
    # regmodel.predict(new_data)
    print(output)
    return jsonify(float(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x)for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The predicted house price is Rs. {}".format(output))

if __name__=="__main__":
    app.run(debug=True)