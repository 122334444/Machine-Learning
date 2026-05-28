import pandas as pd
df=pd.read_csv("carprices.csv")
df.head()

import matplotlib.pyplot as plt

X=df[['Mileage','Age(yrs)']]
Y=df['Sell Price($)']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=10
) #for training it will use 80% of data set and 20% for testing -> will choose 80% of random data

from sklearn.linear_model import LinearRegression
clf=LinearRegression()

clf.fit(X_train,Y_train)

print(clf.predict(X_test))

print(clf.score(X_test,Y_test)) #checks accuracy of model