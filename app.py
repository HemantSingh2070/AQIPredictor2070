from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Define cities list (could be moved to a separate file if needed)
cities = [
    "Agartala", "Agra", "Ahmedabad", "Aizawl", "Ajmer", "Akola", "Alwar", "Amaravati", "Ambala",
    "Amravati", "Amritsar", "Anantapur", "Angul", "Ankleshwar", "Araria", "Ariyalur", "Arrah",
    "Asansol", "Aurangabad", "Aurangabad (Bihar)", "Baddi", "Badlapur", "Bagalkot", "Baghpat",
    "Bahadurgarh", "Balasore", "Ballabgarh", "Banswara", "Baran", "Barbil", "Bareilly", "Baripada",
    "Barmer", "Barrackpore", "Bathinda", "Begusarai", "Belapur", "Belgaum", "Bengaluru", "Bettiah",
    "Bhagalpur", "Bharatpur", "Bhilai", "Bhilwara", "Bhiwadi", "Bhiwandi", "Bhiwani", "Bhopal",
    "Bhubaneswar", "Bidar", "Bihar Sharif", "Bikaner", "Bilaspur", "Bileipada", "Brajrajnagar",
    "Bulandshahr", "Bundi", "Buxar", "Byasanagar", "Byrnihat", "Chamarajanagar", "Chandigarh",
    "Chandrapur", "Charkhi Dadri", "Chengalpattu", "Chennai", "Chhal", "Chhapra", "Chikkaballapur",
    "Chikkamagaluru", "Chittoor", "Chittorgarh", "Churu", "Coimbatore", "Cuddalore", "Cuttack",
    "Damoh", "Darbhanga", "Dausa", "Davanagere", "Dehradun", "Delhi", "Dewas", "Dhanbad", "Dharuhera",
    "Dharwad", "Dholpur", "Dhule", "Dindigul", "Durgapur", "Eloor", "Ernakulam", "Faridabad", "Fatehabad",
    "Firozabad", "Gadag", "GandhiNagar", "Gangtok", "Gaya", "Ghaziabad", "Gorakhpur", "Greater Noida",
    "Gummidipoondi", "Gurugram", "Guwahati", "Gwalior", "Hajipur", "Haldia", "Hanumangarh", "Hapur",
    "Hassan", "Haveri", "Hisar", "Hosur", "Howrah", "Hubballi", "Hyderabad", "Imphal", "Indore",
    "Jabalpur", "Jaipur", "Jaisalmer", "Jalandhar", "Jalgaon", "Jalna", "Jalore", "Jhalawar", "Jhansi",
    "Jharsuguda", "Jhunjhunu", "Jind", "Jodhpur", "Jorapokhar", "Kadapa", "Kaithal", "Kalaburagi",
    "Kalyan", "Kanchipuram", "Kannur", "Kanpur", "Karauli", "Karnal", "Karwar", "Kashipur", "Katihar",
    "Katni", "Keonjhar", "Khanna", "Khurja", "Kishanganj", "Kochi", "Kohima", "Kolar", "Kolhapur",
    "Kolkata", "Kollam", "Koppal", "Korba", "Kota", "Kozhikode", "Kunjemura", "Kurukshetra", "Latur",
    "Loni_Dehat", "Loni_Ghaziabad", "Lucknow", "Ludhiana", "Madikeri", "Mahad", "Maihar", "Mandi Gobindgarh",
    "Mandideep", "Mandikhera", "Manesar", "Mangalore", "Manguraha", "Medikeri", "Meerut", "Milupara",
    "Moradabad", "Motihari", "Mumbai", "Munger", "Muzaffarnagar", "Muzaffarpur", "Mysuru", "Nagaon",
    "Nagaur", "Nagpur", "Naharlagun", "Nalbari", "Nanded", "Nandesari", "Narnaul", "Nashik", "Navi Mumbai",
    "Nayagarh", "Noida", "Ooty", "Pali", "Palkalaiperur", "Palwal", "Panchkula", "Panipat", "Parbhani",
    "Patiala", "Patna", "Pimpri Chinchwad", "Pithampur", "Pratapgarh", "Prayagraj", "Puducherry", "Pune",
    "Purnia", "Raichur", "Raipur", "Rairangpur", "Rajamahendravaram", "Rajgir", "Rajsamand", "Ramanagara",
    "Ramanathapuram", "Ratlam", "Rishikesh", "Rohtak", "Rourkela", "Rupnagar", "Sagar", "Saharsa", "Salem",
    "Samastipur", "Sangli", "Sasaram", "Satna", "Sawai Madhopur", "Shillong", "Shivamogga", "Sikar", "Silchar",
    "Siliguri", "Singrauli", "Sirohi", "Sirsa", "Sivasagar", "Siwan", "Solapur", "Sonipat", "Sri Ganganagar",
    "Srinagar", "Suakati", "Surat", "Talcher", "Tensa", "Thane", "Thiruvananthapuram", "Thoothukudi", "Thrissur",
    "Tiruchirappalli", "Tirupati", "Tirupur", "Tonk", "Tumakuru", "Tumidih", "Udaipur", "Udupi", "Ujjain",
    "Ulhasnagar", "Vapi", "Varanasi", "Vatva", "Vellore", "Vijayapura", "Vijayawada", "Visakhapatnam",
    "Vrindavan", "Yadgir", "Yamunanagar"
]

indexHTML = "index.html"
# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']
        fre = request.form['freq']
        pollutant = request.form['pollutant']

        file_path = f"data/{city}.csv" 
        try:
            air_quality_data = pd.read_csv(file_path)
        except FileNotFoundError:
            return render_template(indexHTML, cities=cities,pollutant=None,city=None, error="File not found for the selected city!")


        # Processing the date column
        date_info = pd.to_datetime(air_quality_data['Date'])
        air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date']).dt.strftime('%Y-%m-%d')
        fixed_time = '12:00:00'
        date_time = pd.concat([date_info, pd.Series([fixed_time] * len(date_info), name='Time')], axis=1)
        date_time['ds'] = date_time['Date'].astype(str) + ' ' + date_time['Time'].astype(str)
        data = pd.DataFrame()
        data['ds'] = pd.to_datetime(date_time['ds'])
        data['y'] = air_quality_data[pollutant]
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=365,freq=fre)
        forecasr = model.predict(future)
        model.plot(forecasr)
        model.plot_components(forecasr)
         # Plot the forecast
        fig1 = model.plot(forecasr)
        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()
        
        # Plot the components
        fig2 = model.plot_components(forecasr)
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()

        return render_template(indexHTML, cities=cities,plot_url=plot_url1,plot_url_01=plot_url2,error=None,pollutant=pollutant,city=city)

    return render_template(indexHTML, cities=cities, plot_url=None,plot_url_01=None,city=None,pollutant=None)

if __name__ == '__main__':
    app.run(debug=True)