# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("hw_25000.csv")

boy = data.Height.values.reshape(-1,1) #gerekli formata dönüştürme işlemi
kilo = data.Weight.values.reshape(-1,1)

regression = LinearRegression()
regression.fit(boy,kilo)    #boy ve kilo için linefit bulur.
print(regression.predict(70)) #verilen değerin tahminlemesini yapar

plt.scatter(data.Height,data.Weight) #x ve y değerleri için grafik oluşturur
plt.plot(boy,regression.predict(boy),color="red") #kırmızı renkte linefit çiz.

plt.xlabel("BOY") #grafiğin x ekseninin ismi
plt.ylabel("KİLO") #grafiğin y eseninin ismi

plt.title("simple regression model") #grafiğin üst kısmında başlık yazmamıza yarar

plt.show()

print(r2_score(kilo,regression.predict(boy))) #algoritmanın başarı oranını bulur.

