from pyexpat import model
import pandas as pd
from flask import Flask, render_template, request,jsonify
import pickle
import instaloader
import numpy as np
import joblib 
from flask import Flask, render_template, request, redirect, url_for
import pymongo


model = joblib.load('mlp_model1.pkl')
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['users']
collection = db['clients']



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('2nd.html')

# @app.route('/fetch', methods=['POST'])
# def fetch():
#     profile_link = request.form['profile_link']
#     username = extract_username(profile_link)
#     user_data = fetch_user_data(username)  # Call the fetch_user_data function
#     return render_template('result.html', user_data=user_data)
@app.route('/predict', methods=['POST'])
def predict():
    profile_link = request.form['profile_link']
    username = extract_username(profile_link)
    features = fetch_user_data(username)
    if features is None:
        return render_template('result.html', prediction_text="Error: Unable to fetch user data.")
    else:
        features_2d = np.array([list(features.values())])  # Convert the dictionary values to a 2D array
        prediction = model.predict_proba(features_2d)[:, 1]
        return render_template("result.html", prediction_text="The possibility of the account being fake is {:.2%}".format(prediction[0]))

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Check if username and password are not empty
    if username and password:
        # Store the username and password in the database
        user_data = {'username': username, 'password': password}
        collection.insert_one(user_data)
        return redirect(url_for('success'))  # Corrected line
    else:
        return render_template('login.html', error='Username and password are required.')

@app.route('/success')
def success():
    return 'Login successful!'
    

def fetch_user_data(username):
    # Create Instaloader instance
    L = instaloader.Instaloader()

    # Login to Instagram
    L.load_session_from_file('ash_cartel52')  # Load session if available
    if not L.context.is_logged_in:
        L.context.login('ash_cartel52', 'ADITYA@4555')  # Replace with your Instagram credentials

    profile = instaloader.Profile.from_username(L.context, username)   
    num_numeric_chars = sum(c.isdigit() for c in username)

    # Calculate the total length of the username
    total_length = len(username)

    # Calculate the ratio of numeric characters to the total length of the username
    if total_length > 0:
        nums_length_ratio = num_numeric_chars / total_length
    else:
        gth_ratio = 0  # Handle division by zero
    
    full_name_words = len(profile.full_name.split())

    # Fetch user data
    user_data = {
            'Profile Pic': 1 if profile.profile_pic_url else 0,
            'Nums/Length Username': nums_length_ratio,
            'Full Name Words' : full_name_words,
            'Bio Length': len(profile.biography),
            'External Url': 1 if profile.external_url else 0,
            'Verified': 1 if profile.is_verified else 0,
            'Business': 1 if profile.is_business_account else 0,
            '#Posts': profile.mediacount,
            '#Followers': profile.followers,
            '#Following': profile.followees,
            # 'username': profile.username,
            # 'username_length': len(profile.username),
            }
    return user_data
def extract_username(profile_link):
    # Assuming the username is the part after 'instagram.com/'
    username_with_slash = profile_link.rsplit('instagram.com/', 1)[-1]
    # Remove trailing slash if present
    if username_with_slash.endswith('/'):
        return username_with_slash[:-1]  # Remove the last character (which is the slash)
    else:
        return username_with_slash
    
    
    
if __name__ == '__main__':
    app.run(debug=True)
