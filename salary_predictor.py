# Salary Prediction using Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

#Load dataset
data = pd.read_csv("Salary_Data.csv")
# simple dataset
a=data["YearsExperience"].values.reshape(-1,1)
b=data["Salary"].values

# create model
model = LinearRegression()
model.fit(a, b)

# checking equation
m = model.coef_[0]
c = model.intercept_

print("Model trained")
print("Equation: Salary =", m, "* Experience +", c)

# take user input
try:
    x = float(input("Enter experience: "))
    pred = model.predict([[x]])
    print("Predicted Salary: ₹", int(pred[0]))
except:
    print("Invalid input")

# plotting
plt.scatter(a,b)
plt.plot(a, model.predict(a))

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction")

plt.show()
