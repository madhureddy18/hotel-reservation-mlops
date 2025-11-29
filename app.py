import joblib
import numpy as np
from config.paths_config import *
from flask import Flask,render_template,request

app = Flask(__name__)

loaded_model=joblib.load(MODEL_OUTPUT_PATH)
@app.route("/",methods=["GET","POST"])
def index():
    prediction = None
    if request.method=="POST":

        lead_time = int(request.form["lead_time"])
        
        avg_price_per_room = float(request.form["avg_price_per_room"])
        no_of_special_requests = int(request.form["no_of_special_requests"])
        arrival_date = int(request.form["arrival_date"])
        arrival_month = int(request.form["arrival_month"])
        market_segment_type = int(request.form["market_segment_type"])
        no_of_week_nights = int(request.form["no_of_week_nights"])
        no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
        no_of_adults = int(request.form["no_of_adults"])
        arrival_year = int(request.form["arrival_year"])
        
        features=np.array([[lead_time,avg_price_per_room,no_of_special_requests,arrival_date,arrival_month,market_segment_type,no_of_week_nights,no_of_weekend_nights,no_of_adults,arrival_year]])

        pred = loaded_model.predict(features)[0]
        prediction = int(pred)

    return render_template('index.html', prediction=prediction)

if __name__=="__main__":
    app.run(host='0.0.0.0' , port=5000)

# ['lead_time' 'avg_price_per_room' 'no_of_special_requests' 'arrival_date'
#  'arrival_month' 'market_segment_type' 'no_of_week_nights'
#  'no_of_weekend_nights' 'no_of_adults' 'arrival_year']