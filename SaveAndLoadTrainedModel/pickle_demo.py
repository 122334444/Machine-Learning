import pandas as pd
import numpy as np
from sklearn import linear_model
import math
import os
import pickle

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "homeprices.csv")

df = pd.read_csv(csv_path)

median_bedrooms = math.floor(df.bedrooms.median())

df.bedrooms = df.bedrooms.fillna(median_bedrooms)

model = linear_model.LinearRegression()

model.fit(df[['area','bedrooms','age']], df.price)

print(model.coef_)

print(model.intercept_)

print(model.predict([[3000,3,40]]))

with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_pickle.pkl', 'rb') as f:
    mp = pickle.load(f)

print(mp.predict([[3000,3,40]]))