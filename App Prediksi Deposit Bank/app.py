from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('bank.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def home():
     try :
        age = request.form['age']
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        ddefault = request.form['default']
        balance = request.form['balance']
        housing = request.form['housing']
        loan = request.form['loan']
        contact = request.form['contact']
        day = request.form['day']
        month = request.form['month']
        duration = request.form['duration']
        campaign = request.form['campaign']
        pdays = request.form['pdays']
        previous = request.form['previous']
        poutcome = request.form['poutcome']

        arr = np.array([[age,job,marital,education,ddefault,balance,housing,loan,
                        contact,day,month,duration, campaign, pdays,previous,poutcome]])

        pred = model.predict(arr)

        return render_template('home.html', data=pred, 
                    age =age, job=job, marital=marital, education=education, ddefault=ddefault, balance=balance, 
                    housing=housing,loan=loan, contact=contact, day=day,month=month,duration=duration, campaign=campaign,
                    pdays=pdays, previous=previous, poutcome=poutcome
            )
        
     except :
        return render_template('home.html')

@app.route('/model')
def model_pred():
    return render_template('model.html')

if __name__ == "__main__":
    app.run(debug=True)
