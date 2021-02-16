import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plp
from statsmodels.stats.weightstats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("diabetes.csv")
dataset.dtypes #Vemos los tipos de datos de la tabla

''' Primero hacemos un boxplot para determinar si podemos intuir
que las personas con diabetes tienen mayor indice de masa corporal que las personas
que no tienen diabetes ''' 
dataset.boxplot(column = ["BMI"], by = ["Outcome"])
''' Las personas con diabetes parecen tener en promedio un mayor indice de
masa corporal. Pero ahora vamos a ver si las diferencias entre promedios
son estadisticamente significativas. Pero primero vamos a ver si hay alguna
relacion entre las variables BMI y diabetes ploteando los puntos'''

plp.plot(dataset["BMI"], dataset["Outcome"], "bo")
''' Pareciera ser que puede haber un aumento en la probabilidad de parecer
diabetes si se aumenta el indice de masa corporal '''

diabetes_positivo = dataset.loc[dataset["Outcome"] == 1]
diabetes_negativo = dataset.loc[dataset["Outcome"] == 0]

ttest_ind(diabetes_positivo["BMI"], diabetes_negativo["BMI"], alternative = "larger")

''' Las diferencias son significativas, mostrando una fuerte evidencia,
en contra de que las personas con diabetes tienen en promedio,
un IMC mayor que las personas que no tienen diabetes. Pero ello
no significa que tener un IMC alto signifique que cause la diabetes,
para ello vamos a construir un modelo logistico que nos va a permitir
sacar esas conclusiones. '''


model = logit(formula = "Outcome ~ Glucose+Pregnancies+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age", data = dataset).fit()

model.summary()
''' Como podemos ver en el reporte del modelo, la variable BMI es estadisticamente
significativa. Eso quiere decir que cambios en el IMC causa cambios en el logaritmo
del ratio de probabilidad, manteniendo constantes variables como la Glucosa, la insulina y
demas factores. Esto nos da un importante aviso de causalidad '''

BMI_coef = list(model.params)[6] #Obtenemos el cambio en el logit de cada coeficiente

np.exp(BMI_coef)

print("Eso significa, que si BMI aumenta en una unidad, el cambio en el logit es:" + str(BMI_coef))

print("Lo podemos interpretar, que un cambio en BMI, multiplica el ratio de probabilidad por: " + str(np.exp(BMI_coef)))

''' Vamos a ver si nos podemos preocupar de problemas de
multicolinealidad con un mapa de calor '''
sn.heatmap(dataset.corr())

''' Pareciera que no debemos preocuparnos de problemas de 
multicolinealidad '''


''' Ahora queremos dar con un modelo que nos permita realizar
predicciones, para saber si una persona tiene o no diabetes '''

X_train, x_test, Y_train, y_test = train_test_split(dataset.drop(["Outcome"], axis = 1), dataset["Outcome"], test_size = 0.33)
''' Primero vamos con el modelo de regresion logistica '''
logit = LogisticRegression()
logit.fit(X_train, Y_train)
predicciones_logit = logit.predict(x_test)

accuracy_score(y_true = y_test, y_pred = predicciones_logit)



''' Ahora vamos a entrenar un arbol de decision para clasificacion '''

arbol = DecisionTreeClassifier(max_depth = 5)
arbol.fit(X_train, Y_train)

''' Vamos a ver la cantidad de nodos terminales 
del arbol '''
arbol.get_n_leaves()

plot_tree(arbol, feature_names = list(dataset.drop(["Outcome"], axis = 1).columns), class_names = "Diabetes", filled = True, fontsize = 7)

predicciones_arbol = arbol.predict(X = x_test)
accuracy_score(y_true = y_test, y_pred = predicciones_arbol)

''' Parece ser que el modelo logistico tiene un mayor nivel de predictibilidad 
que el modelo de arbol a pesar de que restringimos restringir la profundidad
del arbol. Esto se puede deber a distintos factores, siendo uno de los mas importantes
el tamaño de entrenamiento y prueba de la muestra elegidos. Para evitar eso
vamos a considerar varios tamaños de entrenamiento y prueba '''

accuracy_arbol = []
accuracy_logit = []

for i in np.linspace(0.5, 0.95, num = 45):
    X_train, x_test, Y_train, y_test = train_test_split(dataset.drop(["Outcome"], axis = 1), dataset["Outcome"], test_size = i)
    arbol = DecisionTreeClassifier(max_depth = 5)
    arbol.fit(X_train, Y_train)
    predicciones_arbol = arbol.predict(X = x_test)
    accuracy_arbol.append(accuracy_score(y_true = y_test, y_pred = predicciones_arbol))
    
for k in np.linspace(0.5, 0.95, num = 45):
    X_train, x_test, Y_train, y_test = train_test_split(dataset.drop(["Outcome"], axis = 1), dataset["Outcome"], test_size = k)
    logit = LogisticRegression()
    logit.fit(X_train, Y_train)
    predicciones_logit = logit.predict(x_test)
    accuracy_logit.append(accuracy_score(y_true = y_test, y_pred = predicciones_logit))
''' Trazamos histogramas de los accuracy de ambos modelos '''
fig, ax = plp.subplots(2, 1)
ax[0].hist(accuracy_arbol, color = "red")
ax[0].set_xlabel("Accuracy arbol")

ax[1].hist(accuracy_logit, color = "orange")
ax[1].set_xlabel("Accuracy logit")
plp.show()

''' Pareciera ser que el modelo logistico tiene mayor nivel de predicciones. Vamos
a ver si las diferencias son significativas '''

ttest_ind(accuracy_logit, accuracy_arbol, alternative = "larger")

''' Entonces nos quedamos con el modelo de regresion logistica '''

