#!/usr/bin/env python
import numpy as np
import pandas as pd
import time

# Data Importing

data = pd.read_csv("Datasets/Admission_Predict_Ver1.1.csv")

# Linear Regression Model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model = LinearRegression()

y = data['Chance of Admit '].values.reshape(-1, 1)
X = data.drop(['Chance of Admit ', 'Serial No.'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results

print("Root Mean Squared Error: {}\n\n".format(rmse))

# Graph results

# import matplotlib.pyplot as plt

# plt.scatter(X, y)
# plt.plot(X_test, y_pred, color='black', linewidth=3)
# plt.show()

Xnew = np.array([[335, 105, 4, 4, 5, 8.2, 1]])
ynew = model.predict(Xnew)
print("\nX=%s\n Predicted=%s\n\n" % (Xnew[0], ynew[0]))

# while(1):
#     gre = int(input("What is your GRE Score? (0-350)\n"))
#     toefl = int(input("TOEFL Score? (0-120)\n"))
#     uni_rating = int(input("University Rating (1-5)\n"))
#     sop = float(input("Statement of Purpose strength (1-5)\n"))
#     lor = float(input("Letter of Recommendation strength (1-5)\n"))
#     gpa = float(input("Undergraduate GPA? (4 point system)\n")) * 2.5
#     res_exp = int(input("Research Experience?\n1: True, 0: False\n"))
#
#     out = model.predict(
#         np.array([[gre, toefl, uni_rating, sop, lor, gpa, res_exp], ]))
#     print("Your probability of acceptance is {}".format(out))
#     time.sleep(3)
