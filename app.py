from flask import Flask, render_template, request, redirect, url_for, flash, session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from functools import wraps
import os
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = 'trial'
# Create users.json if it doesn't exist
if not os.path.exists('users.json'):
    with open('users.json', 'w') as f:
        json.dump({}, f)

# Load the pre-trained models
model = load_model('./models/cnn.h5')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        with open('users.json', 'r') as f:
            users = json.load(f)
            
        if username in users and users[username]['password'] == password:
            session['username'] = username
            flash('Successfully logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
            
        with open('users.json', 'r') as f:
            users = json.load(f)
            
        if username in users:
            flash('Username already exists', 'error')
            return render_template('signup.html')
            
        users[username] = {
            'password': password
        }
        
        with open('users.json', 'w') as f:
            json.dump(users, f)
            
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Successfully logged out!', 'success')
    return redirect(url_for('home'))

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    prediction = None
    plot_data = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return render_template('detect.html')
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return render_template('detect.html')
            
        if file and file.filename.endswith('.csv'):
            try:
                features = pd.read_csv(file)
                time = [i / 178 for i in range(178)]
                plt.figure(figsize=(10, 6))
                plt.plot(time, features.iloc[0], color='#3498db')
                plt.xlabel('Time (seconds)')
                plt.ylabel('EEG Amplitude')
                plt.title('EEG Signal Over 1 Second')
                plt.grid(True)
                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                prediction = model.predict(features.values)[0]
                prediction = (prediction > 0.5).astype(int)  # Convert to 0 or 1
                print(prediction)
                prediction = "Detected Epileptic Seizure" if prediction==1 else "No Epileptic Seizure Detected"
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
        else:
            flash('Please upload a CSV file', 'error')
            
    return render_template('detect.html', prediction=prediction, plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)