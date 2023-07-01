


#Regresion Lineal Simple.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Dividir  el conjunto data set en conjunto  de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state= 0)

#Crear modelo de Regresion Lineal con el conjunto de entrenamiento.

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test.
y_pred = regression.predict(X_test)

#Visualizar los resultados de entrenamientos
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs años de  experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en ($)")
plt.show()


#Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs años de  experiencia (Conjunto de Prueba)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en ($)")
plt.show()