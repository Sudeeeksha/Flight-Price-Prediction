from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("rf_reg.pkl", "rb"))

@app.route("/")
def index():
    return render_template("predict.html")

@app.route("/flight_predict",methods=['GET','POST'])
def flight_predict():
    if request.method == 'GET':
        return render_template('predict.html')  # Or any default page
    if request.method == 'POST':
        # departure
        date_dep = request.form["departure_date"]
        if "T" not in date_dep:
          date_dep += "T00:00"  # Default to midnight
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep,format='%Y-%m-%dT%H:%M').month)
        
        #departure time
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)

        #arrival
        date_ar = request.form["arrival_date"]
        Arrival_hour = int(pd.to_datetime(date_ar, format="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_ar,format='%Y-%m-%dT%H:%M').minute)

        #duration
        Duration_hours = abs(Arrival_hour-Dep_hour)
        Duration_mins = abs(Arrival_min-Dep_min)


        #airline
        airline = request.form['airline']

        # Define airline mapping based on HTML options
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

        # Set the selected airline to 1
        if airline in airlines:
            airline_vars[airlines[airline]] = 1

        #Total stops
        Total_Stops = int(request.form["stopage"])

        # Get source from form
        source = request.form["source"]
        # Define source mapping
        sources = {
        
            "Delhi": "Source_Delhi",
            "Mumbai": "Source_Mumbai",
            "Kolkata": "Source_Kolkata",
            "Chennai": "Source_Chennai"
        }
        # Initialize all source variables to 0
        source_vars = {key: 0 for key in sources.values()}
        # Set the selected source to 1
        if source in sources:
            source_vars[sources[source]] = 1

        # Get destination from form
        destination = request.form["destination"]
        # Define destination mapping
        destinations = {
            "New Delhi": "Destination_New Delhi",
            "Delhi": "Destination_Delhi",
            "Cochin": "Destination_Cochin",
            "Hyderabad": "Destination_Hyderabad",
            "Kolkata": "Destination_Kolkata"
        }
        # Initialize all destination variables to 0
        destination_vars = {key: 0 for key in destinations.values()}
        # Set the selected destination to 1
        if destination in destinations:
            destination_vars[destinations[destination]] = 1

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
            *airline_vars.values(),   # Airline variables
            *source_vars.values(),    # Source variables
            *destination_vars.values() # Destination variables
        ]])
        
    output=round(prediction[0],2)

    return render_template('predict.html',prediction_text="Your Flight price is Rs. {}".format(output))


if __name__ == "__main__":
    app.run(debug=True) 
