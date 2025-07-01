from flask import Flask, request, render_template, jsonify
import sklearn
import pickle
import pandas as pd
from datetime import datetime, timedelta
import json

app = Flask(__name__)
model = pickle.load(open("rf_reg.pkl", "rb"))

@app.route("/")
def index():
    return render_template("predict.html")

@app.route("/flight_predict", methods=['POST'])
def flight_predict():
    try:
        # Get form data
        departure_date = request.form["departure_date"]
        departure_time = request.form["departure_time"]
        duration_hours = int(request.form["duration_hours"])
        duration_minutes = int(request.form["duration_minutes"])
        
        # Combine departure date and time
        departure_datetime_str = f"{departure_date}T{departure_time}"
        departure_datetime = pd.to_datetime(departure_datetime_str, format="%Y-%m-%dT%H:%M")
        
        # Calculate arrival datetime
        duration_total_minutes = duration_hours * 60 + duration_minutes
        arrival_datetime = departure_datetime + timedelta(minutes=duration_total_minutes)
        
        # Extract features
        Journey_day = departure_datetime.day
        Journey_month = departure_datetime.month
        Dep_hour = departure_datetime.hour
        Dep_min = departure_datetime.minute
        Arrival_hour = arrival_datetime.hour
        Arrival_min = arrival_datetime.minute
        Duration_hours = duration_hours
        Duration_mins = duration_minutes

        # Get other form data
        airline = request.form['airline']
        source = request.form["source"]
        destination = request.form["destination"]
        Total_Stops = int(request.form["stopage"])

        # Validate that source and destination are different
        if source == destination:
            return jsonify({
                'success': False, 
                'error': 'Source and destination cannot be the same'
            })

        # Define airline mapping
        airlines = {
            "Jet Airways": "Airline_Jet Airways",
            "IndiGo": "Airline_IndiGo",
            "Air India": "Airline_Air India",
            "Multiple carriers": "Airline_Multiple carriers",
            "SpiceJet": "Airline_SpiceJet",
            "Vistara": "Airline_Vistara",
            "GoAir": "Airline_GoAir",
            "Multiple carriers Premium economy": "Airline_Multiple carriers Premium economy",
            "Jet Airways Business": "Airline_Jet Airways Business",
            "Vistara Premium economy": "Airline_Vistara Premium economy",
            "Trujet": "Airline_Trujet"
        }

        # Initialize all airline variables to 0
        airline_vars = {key: 0 for key in airlines.values()}
        if airline in airlines:
            airline_vars[airlines[airline]] = 1

        # Define source mapping
        sources = {
            "Delhi": "Source_Delhi",
            "Mumbai": "Source_Mumbai",
            "Kolkata": "Source_Kolkata",
            "Chennai": "Source_Chennai"
        }
        source_vars = {key: 0 for key in sources.values()}
        if source in sources:
            source_vars[sources[source]] = 1

        # Define destination mapping
        destinations = {
            "New Delhi": "Destination_New Delhi",
            "Delhi": "Destination_Delhi",
            "Cochin": "Destination_Cochin",
            "Hyderabad": "Destination_Hyderabad",
            "Kolkata": "Destination_Kolkata"
        }
        destination_vars = {key: 0 for key in destinations.values()}
        if destination in destinations:
            destination_vars[destinations[destination]] = 1

        # Make prediction
        prediction = model.predict([[
            Total_Stops,
            Journey_day,
            Journey_month,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            Duration_hours,
            Duration_mins,
            *airline_vars.values(),
            *source_vars.values(),
            *destination_vars.values()
        ]])
        
        output = round(prediction[0], 2)
        
        # Format the result with flight details
        result = {
            'success': True,
            'price': output,
            'departure_datetime': departure_datetime.strftime("%Y-%m-%d %H:%M"),
            'arrival_datetime': arrival_datetime.strftime("%Y-%m-%d %H:%M"),
            'duration': f"{duration_hours}h {duration_minutes}m",
            'route': f"{source} → {destination}",
            'airline': airline,
            'stops': Total_Stops
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)