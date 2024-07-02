import requests
import customtkinter as ctk
from tkinter import messagebox, Toplevel
from PIL import Image, ImageTk
import geocoder
from plyer import notification
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyttsx3 
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
# Load the weather prediction model
model = load_model('weather_model.h5')
# Functions to get weather data
def get_weather(city_name, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric'
    response = requests.get(url)
    print(f"Weather API response: {response.json()}")  # Log the response
    return response.json()

def get_forecast(city_name, api_key):
    url = f'http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}&units=metric'
    response = requests.get(url)
    print(f"Forecast API response: {response.json()}")  # Log the response
    return response.json()

def get_aqi(lat, lon, api_key):
    url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}'
    response = requests.get(url)
    print(f"AQI API response: {response.json()}")  # Log the response
    return response.json()

def get_location():
    g = geocoder.ip('me')
    return g.city

# Functions to handle image capturing and prediction
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_image.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match model input size
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_weather_from_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    classes = ['Clear', 'Clouds', 'Rain', 'Snow']  # Adjust based on your model's classes
    return classes[class_index]

# Functions to display weather data and notifications
def show_weather():
    threading.Thread(target=_show_weather).start()

def _show_weather():
    city_name = city_entry.get()
    weather_api_key = 'c773d799ee81fdd65e03ad56640278dc'
    
    weather_data = get_weather(city_name, weather_api_key)
    forecast_data = get_forecast(city_name, weather_api_key)
    
    if weather_data['cod'] == 200:
        lat = weather_data['coord']['lat']
        lon = weather_data['coord']['lon']
        aqi_data = get_aqi(lat, lon, weather_api_key)

        if 'list' in aqi_data:
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            wind_speed = weather_data['wind']['speed']
            weather_condition = weather_data['weather'][0]['main']
            
            aqi = aqi_data['list'][0]['main']['aqi']
            aqi_advice = get_aqi_advice(aqi)
            
            # Capture and process image for weather prediction
            capture_image()
            image_weather_condition = predict_weather_from_image('captured_image.jpg')
            
            # Reminder and rain prediction based on weather condition and humidity
            reminder = ""
            rain_prediction = ""
            if image_weather_condition.lower() in ["rain", "drizzle", "thunderstorm"]:
                reminder = "Don't forget to take an umbrella!"
                rain_prediction = "High chance of rain today."
                if reminder_toggle.get() == 1:
                    print("Sending rain notification...")
                    send_notification("Weather Alert", "Chance of rain in your area within 1 hour. Take an umbrella!")
            elif humidity > 80:  # Threshold for high humidity indicating potential rain
                rain_prediction = "High humidity detected. There might be a chance of rain today."
                if reminder_toggle.get() == 1:
                    print("Sending humidity notification...")
                    send_notification("Humidity Alert", "High humidity detected. There might be a chance of rain today.")
            else:
                rain_prediction = "No significant chance of rain today. You may go outing."

            result = (f"Weather in {city_name}:\n"
                      f"Temperature: {temperature}°C\n"
                      f"Humidity: {humidity}%\n"
                      f"Wind Speed: {wind_speed} m/s\n\n"
                      f"Condition: {weather_condition}\n"
                      f"Image Predicted Condition: {image_weather_condition}\n\n"
                      f"Air Quality Index (AQI): {aqi}\n"
                      f"{aqi_advice}\n\n"
                      f"{reminder}\n"
                      f"{rain_prediction}")
            
            # Update the weather image
            update_weather_image(weather_condition)
            
            # Display the weather information
            messagebox.showinfo("Weather Information", result)
            
            # Read aloud the weather information
            read_aloud(result)
            
            # Display the weather forecast
            show_forecast(forecast_data)
        else:
            messagebox.showerror("Error", f"AQI API Error: {aqi_data['message']}")
    else:
        messagebox.showerror("Error", f"Weather API Error: {weather_data['message']}")

def get_aqi_advice(aqi):
    if aqi == 1:
        return "Air quality is good."
    elif aqi == 2:
        return "Air quality is fair."
    elif aqi == 3:
        return "Air quality is moderate."
    elif aqi == 4:
        return "Air quality is poor."
    elif aqi == 5:
        return "Air quality is very poor."
    else:
        return "Air quality data not available."

def update_weather_image(condition):
    img_path = ""
    if condition.lower() == "clear":
        img_path = "images/sunny.png"
    elif condition.lower() == "rain":
        img_path = "images/rainy.png"
    elif condition.lower() == "clouds":
        img_path = "images/cloudy.png"
    elif condition.lower() == "snow":
        img_path = "images/snowy.png"
    else:
        img_path = "images/default.png"
    
    img = Image.open(img_path)
    img = img.resize((100, 100), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    weather_image_label.configure(image=photo)
    weather_image_label.image = photo  # Keep a reference to avoid garbage collection

def send_notification(title, message):
    try:
        print(f"Sending notification: {title} - {message}")  # Log the notification
        notification.notify(
            title=title,
            message=message,
            app_name="Weather App",
            timeout=10
        )
        print("Notification sent successfully")
    except Exception as e:
        print(f"Failed to send notification: {e}")

def show_forecast(forecast_data):
    # Extract forecast data for the next 3 days
    three_day_forecast = forecast_data['list'][:24*3:8]  # Every 8th item is 24 hours later
    
    dates = [item['dt_txt'] for item in three_day_forecast]
    temps = [item['main']['temp'] for item in three_day_forecast]
    weather_descriptions = [item['weather'][0]['description'] for item in three_day_forecast]
    
    # Create a new window for the forecast
    forecast_window = Toplevel(root)
    forecast_window.title("3-Day Weather Forecast")
    forecast_window.geometry("800x600")
    
    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dates, temps, marker='o', label='Temperature')
    
    # Format the plot
    ax.set_title(f"3-Day Temperature Forecast for {city_entry.get()}")
    ax.set_xlabel("Date and Time")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)
    plt.xticks(rotation=45, ha='right')
    
    for i, (date, temp, desc) in enumerate(zip(dates, temps, weather_descriptions)):
        ax.text(date, temp, f"{desc}\n{temp}°C", fontsize=9, ha='center')
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=forecast_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def read_aloud(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Create the main window
root = ctk.CTk()
root.title("Weather App")
root.geometry("600x600")

# Create and place the widgets
ctk.CTkLabel(root, text="Enter city name:").pack(pady=10)

city_entry = ctk.CTkEntry(root)
city_entry.pack(pady=5)

ctk.CTkButton(root, text="Get Weather", command=show_weather).pack(pady=20)

# Placeholder for weather image
weather_image_label = ctk.CTkLabel(root, text="")
weather_image_label.pack(pady=10)

# Reminder toggle
reminder_toggle = ctk.IntVar()
ctk.CTkCheckBox(root, text="Enable Weather Reminders", variable=reminder_toggle).pack(pady=10)

# Add a button to detect current location
def detect_location():
    current_city = get_location()
    city_entry.delete(0, 'end')
    city_entry.insert(0, current_city)

ctk.CTkButton(root, text="Use Current Location", command=detect_location).pack(pady=10)

# Add a button to capture image and get weather
ctk.CTkButton(root, text="Capture Image & Get Weather", command=show_weather).pack(pady=20)

# Start the GUI event loop
root.mainloop()
