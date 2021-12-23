# -*- coding: utf-8 -*-
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pandas.read_csv("positions.csv")

level = data.Level.values.reshape(-1,1)
salary = data.Salary.values.reshape(-1,1)

regression = DecisionTreeRegressor()
regression.fit(level,salary)

plt.scatter(level,salary,color="red")
x = np.arange(min(level),max(level),0.01).reshape(-1,1) # en önemli satır burası.
plt.plot(x,regression.predict(x),color="blue")
plt.xlabel("level")
plt.ylabel("salary")
plt.title("decision tree model")
plt.show()

# np.arange fonksiyonunda min maxtan sonra ki artış miktarını vermezsek istediğimiz
# modele ulaşamayız !!!