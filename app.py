# importing the required libraries
from flask import Flask, render_template, request, redirect, url_for
from joblib import load
from preprocess_flask import preprocess
import pandas as pd

# load the pipeline object
pipeline = load("paygo_default.joblib")

# start flask
app = Flask(__name__)

### Webpage still a work in progress

# render default webpage
@app.route('/')
def home():
    return render_template('paygo.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        df_path = str(request.form['search'])
        df =pd.read_csv(df_path,index_col=('ID') ).iloc[:3,:]
        # get the prediction
        cols = ['m1','m2','m3','m4','m5','m6']
        df[cols] = pipeline.predict(df)
        df_result = df[cols]       
        return render_template('paygo.html', tables=[df_result.to_html(classes='data')], titles=df_result.columns.values )
        

if __name__ == '__main__':
    app.run(debug=True)