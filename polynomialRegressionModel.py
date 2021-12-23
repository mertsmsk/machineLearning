# -*- coding: utf-8 -*-

import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pandas.read_csv("positions.csv")

level = data.Level.values.reshape(-1,1)
salary = data.Salary.values.reshape(-1,1)

# lineer regression
regression = LinearRegression()
regression.fit(level,salary)

tahminLineer = regression.predict(8.3) #lineer çizimdeki tahmini sonuç

# polynomial regression
regressionPoly = PolynomialFeatures(degree = 4) #en düzgün grafik için 4 değeri verilebilir.
levelPoly = regressionPoly.fit_transform(level) #level değerlerine bakarak bunları polinomal şekle sok.
regression2 = LinearRegression()
regression2.fit(levelPoly,salary)

tahminPoly = regression2.predict(regressionPoly.fit_transform(8.3)) # polinomal çizimdeki tahmini sonuç

plt.scatter(level,salary,color="blue")

plt.plot(level,regression.predict(level),color="red") #lineer çizim
plt.plot(level,regression2.predict(levelPoly),color="red") #polynomial çizim

plt.xlabel("LEVEL")
plt.ylabel("SALARY")
plt.title("Lineer And Polynomial Graphic")
plt.show()
