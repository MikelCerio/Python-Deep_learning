# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:07:30 2023

@author: Mikel
"""

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
from scipy.io import arff




from numpy.random import seed
seed(1)




# Cargamos el conjunto de datos
os.chdir(r"C:\Users\PCUser\Desktop\certificado google data analytics\DEEP LEARNING\Módulo 7\Examen")
df = arff.loadarff(r"dataset_37_diabetes.arff")

df = pd.DataFrame(df[0])

df.head()

df['class2'] = pd.get_dummies(df['class'],drop_first=True)

df['class'][df['class2']==True]=1
df['class'][df['class2']==False]=0
df = df.drop('class2', axis=1)

df['class']=df['class'].astype(int)

X = df.iloc[:,0:8]
y = df.iloc[:,8]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        train_size   = 0.7,
                                        random_state = 123,
                                        shuffle      = True
                                    )


# ### Definimos el modelo Keras




# Para obtener todos el mismo resultado debemos añadir una semilla
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)
seed(1)





# Definiremos el modelo como una secuencia de capas.
# Usaremos el modelo secuencial de manera que podamos ir añadiendo capas hasta estar contentos con la arquitectura desarrollada.
model = Sequential()

# Partimos de un sistema con 8 variables por lo que nuestra primera capa acomodará dichas variables
# En la primera capa oculta usaremos 12 neuronas y una función de activación ReLU
# En la segunda capa oculta usaremos 8 neuronas y una función de activación ReLU
# Finalmente en la de salida una neurona y función de activación sigmoide
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))




# ### Compilamos el modelo


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              #loss="binary_crossentropy",
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, # usar cuando valores de predicción sean [0,1]
                                                      label_smoothing=0.0,
                                                      axis=-1,
                                                      reduction="auto",
                                                      name="binary_crossentropy"),
              metrics=['accuracy'])


                        drop_first=True)
model.fit(X_train, y_train, epochs=300, batch_size=20)


# Con la red neuronal entrenada, ahora debemos evaluar cómo ha funcionado.
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


predicciones = model.predict(X_test)

y_pred = (predicciones > 0.5).astype(int)
y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

