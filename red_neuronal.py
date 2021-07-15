# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:01:25 2021

@author: carlos lamas rodriguez carlos.lamas.rodriguez@rai.usc.es
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

plt.close()
datos=pd.read_excel('datos_pandas2.xlsx')
datos=datos.fillna(0)
#print(datos)

# datos=datos.drop(['Ntele', 'NtMu'], axis=1)
# #print(datos)
# 'NCElR1', 'NCElR2', 'NCElR3', 'NCElR4', 'NCMuR1', 'NCMuR2', 'NCMuR3', 'NCMuR4', 'NCMxR1', 'NCMxR2', 'NCMxR3', 'NCMxR4'

# corr=datos.corr()
# #print(corr)
# plt.figure(figsize=(15, 15))
# sns.heatmap(corr, cmap='RdBu', vmax=1, vmin=-1) #Representamos la matriz de correlación
# plt.title('Matriz de correlación')
# #np.savetxt('Matriz de correlación', corr)

# corr_E=pd.DataFrame(corr,columns=['EnePCR'])
# corr_PrimCR=pd.DataFrame(corr,columns=['PrimCR'])

# plt.figure()
# plt.plot(corr_PrimCR); plt.plot(corr_PrimCR, '*'); plt.xticks(rotation=90); plt.hlines(0,'PrimCR', 'NCMx/NCEl', colors='black'); plt.title('Correlación tipo de núcleo')

# plt.figure()
# plt.plot(corr_E); plt.plot(corr_E, '*'); plt.xticks(rotation=90); plt.hlines(0,'PrimCR', 'NCMx/NCEl', colors='black'); plt.title('Correlación E PCR')


datos_barajados=datos.sample(frac=1).reset_index(drop=True) #Reordenamos las filas para que los datos de test abarquen todo el rango
train=datos_barajados[3:]
test=datos_barajados[0:3]
resultados_test=pd.DataFrame(test,columns=['PrimCR', 'EnePCR'])
test=test.drop(['PrimCR', 'EnePCR'],axis=1)
#print(train)
#print(test)

X=train.drop(['PrimCR', 'EnePCR'],axis=1).values
Y=pd.DataFrame(train,columns=['PrimCR', 'EnePCR']).values
#print (Y)

assert X.shape[0]==Y.shape[0]
input_dim=X.shape[1]
output_dim=Y.shape[1]

model=keras.models.Sequential([keras.layers.Dense(100,activation=tf.nn.relu,input_shape=(input_dim,)),keras.layers.Dense(100,activation=tf.nn.relu), keras.layers.Dense(output_dim)])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])
model.summary()

validation_split=0.2
history=model.fit(X,Y,workers=4,epochs=100,verbose=2,validation_split=validation_split)

errores=pd.DataFrame(history.history)
f=plt.figure(figsize=(20,10))
rows=1; cols=2
ax=f.add_subplot(rows,cols,1)
sns.lineplot(data=errores[['mean_absolute_percentage_error', 'val_mean_absolute_percentage_error']].iloc[3:-1])
ax = f.add_subplot(rows, cols, 2)
sns.lineplot(data=errores[["loss", "val_loss"]].iloc[3:-1])
#f.savefig('Errores')

prediction=model.predict(test.values)
resultados=pd.DataFrame(prediction, columns=['PrimCR', 'EnePCR'])

print(resultados_test)
resultados_test.head()
print(resultados)