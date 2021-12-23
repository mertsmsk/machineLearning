# -*- coding: utf-8 -*-

import pandas
from sklearn.ensemble import RandomForestRegressor

data = pandas.read_csv("positions.csv")

level = data.Level.values.reshape(-1,1)
salary = data.Salary.values

regression = RandomForestRegressor(n_estimators=5, random_state=0) # 5 tane dallanma olsun demektir.
regression.fit(level,salary)

print(regression.predict(8.2))