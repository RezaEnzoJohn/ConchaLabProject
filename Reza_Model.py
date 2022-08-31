from joblib import Parallel, delayed
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import sys, os
import time


def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



with open(resource_path("C:/Users/gjaha/pyproj/Take_Home_Project_ Machine_Learning/RandomForest1.pkl"), "rb") as f:
    forest1 = joblib.load(f)
with open(resource_path("C:/Users/gjaha/pyproj/Take_Home_Project_ Machine_Learning/RandomForest2.pkl"), "rb") as f1:
    forest2 = joblib.load(f1)
with open(resource_path("C:/Users/gjaha/pyproj/Take_Home_Project_ Machine_Learning/RandomForest3.pkl"), "rb") as f2:
    forest3 = joblib.load(f2)
with open(resource_path("C:/Users/gjaha/pyproj/Take_Home_Project_ Machine_Learning/RandomForest4.pkl"), "rb") as f3:
    forest4 = joblib.load(f3)
    
val_2k = int(input("Enter individuals hearing threshold gain value (in dB) at 2kHz: "))
val_4k = int(input("Enter individuals hearing threshold gain value (in dB) at 4kHz: "))
val_6k = int(input("Enter individuals hearing threshold gain value (in dB) at 6kHz: "))

inputs1 = np.array([val_2k, val_4k, val_6k], dtype='int8')
out1 = forest1.predict([inputs1]).astype(int)
inputs2 = np.array([val_2k, val_4k, val_6k, out1[0]], dtype='int8')
out2 = forest2.predict([inputs2]).astype(int)
inputs3 = np.array([val_2k, val_4k, val_6k, out1[0], out2[0]], dtype='int8')
out3 = forest3.predict([inputs3]).astype(int)
inputs4 = np.array([val_2k, val_4k, val_6k, out1[0], out2[0], out3[0]], dtype='int8')
out4 = forest4.predict([inputs4]).astype(int)
output = np.array([val_2k, val_4k, val_6k, out1[0], out2[0], out3[0], out4[0]], dtype='int8')

print()
print()

print("Individual's hearing threshold gain values are as follows:")
print("500Hz:", output[6], "(predicted)")
print("1kHz:", output[5], "(predicted)")
print("2kHz:", output[0])
print("3kHz:", output[3], "(predicted)")
print("4kHz:", output[1])
print("6kHz:", output[2])
print("8kHz:", output[4], "(predicted)")

time.sleep(30)
