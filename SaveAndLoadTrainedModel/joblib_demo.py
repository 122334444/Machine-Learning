import pandas as pd
import numpy as np
from sklearn import linear_model
import math
import os
import joblib

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

joblib.dump(model, 'model_joblib')

mj = joblib.load('model_joblib')

print(mj.predict([[3000, 3, 40]]))