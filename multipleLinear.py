# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("insurance.csv")

expenses = data.expenses.values.reshape(-1,1) #y ekseni olsun.
ageBmi = data.iloc[:,[0,2]].values #x ekseni olsun.

regression = LinearRegression()
regression.fit(ageBmi,expenses)

print(regression.predict([[20,20]]))

#20 yaşında ve 20 bmi sahip birinin harcamaları ortalaması.