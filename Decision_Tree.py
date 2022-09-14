# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 00:34:26 2020

@author: monte
"""

import numpy as np 
import pandas as pd


from sklearn.tree import DecisionTreeClassifier


#Cargamos la base de datos
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]

#Definimos las caracteristicas del dataset
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

"""algunas características son de categoría,
 tales como Sex o__BP__. Desafortunadamente, 
 los árboles de Decisión Sklearn no manejan variables categóricas.
 Pero las podemos convertir en valores numéricos.
 pandas.get_dummies() Convertir variable categórica 
 en indicadores de variables."""
 #conversion a valores numericas para las categorias no numericas.
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 
X[0:5]



#Establecemos la variable objetivo, es decir se minizara la entropia para la columna "drogas"
y = my_data["Drug"]
y[0:5]

#entrenar/probar separar en nuestro árbol de decisión.
#La X e y son los arreglos necesarios ántes de la operación dividir/separar, 
#test_size representa el grado del dataset de pruebas, y el random_state asegura que obtendremos las mismas divisiones. 
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


#Primero crearemos una instancia del DecisionTreeClassifier llamada drugTree.
#Dentro del clasificador, especificaremos criterion="entropy" para que podamos ver la nueva información de cada nodo.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # muestra los parámetros por omisión
drugTree.fit(X_trainset,y_trainset)



predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])


#EVALUACION DEL MODELO 
from sklearn import metrics
import matplotlib.pyplot as plt
print("Precisión de los Arboles de Decisión: ", metrics.accuracy_score(y_testset, predTree))


#Visualizacion
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')