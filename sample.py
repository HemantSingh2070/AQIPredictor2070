from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
import os
file_path = "data/Agartala.csv" 
air_quality_data = pd.read_csv(file_path)
air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date'], errors='coerce', dayfirst=True)
air_quality_data['time'] = "00:00:00"
air_quality_data['ds'] = air_quality_data['Date'].astype(str)+ air_quality_data['time']
data = pd.DataFrame()
data['ds'] = air_quality_data['ds']
data['y'] = air_quality_data['AQI']
print(data)