#!/usr/bin/env python
# coding: utf-8

# # Actividad Final MIA05

# Integrantes:
# 
# Juan Carlos Muñoz Esquivel.
# María Delina Culebro Farrera.
# Aline Hernandez Garcia.
# Miguel Ángel Tamer Meyer.
# Aurora Correa Flores.
# 

# Librerías

# In[146]:


import torch
torch.__version__


# In[147]:


import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Importando conjunto de datos.

# In[148]:


#cargando conjuntos de datos
breast = datasets.load_breast_cancer()
#ver lo datos
breast.data


# In[149]:


#nombre de las columnas
breast.feature_names


# In[150]:


#Para observar los objetivos tenemos dos clases 0 y 1
breast.target


# In[151]:


#nombres de las clases
breast.target_names


# In[152]:


inputs = breast.data
#569 instancias y 30 columnas
inputs.shape


# In[153]:


outputs = breast.target
#569 intancias
outputs.shape


# In[154]:


#partir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.25)


# In[155]:


X_train.shape


# In[156]:


#25% de los datos que vamos a probar
X_test.shape


# Transformación de los datos

# In[157]:


type(X_train)
# Se necesita cambiar de numpy.ndarray a pytorch Tensor


# In[158]:


X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)


# In[159]:


type(X_train)


# In[160]:


# Se convierte también el dataset
dataset = torch.utils.data.TensorDataset(X_train,y_train, )


# In[161]:


type(dataset)


# In[162]:


train_loader = torch.utils.data.DataLoader(dataset, batch_size = 10)


# Structura de la Red Neuronal 

# In[163]:


#(Inputs+Ouputs)/2 -> (30+1)/2 = 15.5 = 16
#30 -> 16 -> 16 -> 1
network = nn.Sequential(nn.Linear(in_features=30,out_features=16),
                        nn.Sigmoid(),
                        nn.Linear(16, 16),
                        nn.Sigmoid(),
                        nn.Linear(16, 1),
                        nn.Sigmoid())



# In[164]:


network.parameters


# In[165]:


# Si tenemos un error binario podemos usar BCELoss Binary Cross Entropy para calcularlo
loss_function = nn.BCELoss() 


# In[166]:


#Creación del optimizador que actualiza los pesos y la tasa de aprendizaje
optimizer = torch.optim.Adam(network.parameters(), lr = 0.001) 


# Entrenamiento

# In[167]:


epochs =100
for epoch in range(epochs):
    running_loss= 0. # Está variable guarda el valor en orden para imprimir después de cada uno de los epochs
    
    #ir para cada instancias
    for data in train_loader:
        inputs, outputs = data #crea entradas y salidas
        #print(inputs)
        #print('-----')
        #print(outputs)
        #reset el Gradiante después de cada grupo de 10 instancias
        optimizer.zero_grad() 
        
        #para las predicciones de esto toma cada grupo y la red neuronal va aplicar la función suma, la activación de la función, y optenemos la predicción en la capa de salida 
        predictions = network.forward(inputs)
        loss = loss_function(predictions.squeeze(), outputs) #removiendo la dimension extra de las salidas esperadas del conjunto de datos
        loss.backward() #función para la propagación de hacia atrás
        optimizer.step() # Actualización de los pesos
        
        #Aquí se calcula el error por cada uno de los grupos
        running_loss += loss.item()
        #Se imprimen los resultados en orden de obtener un promedio
    print('Epoch: ' + str(epoch + 1) + ' loss: ' + str(running_loss / len(train_loader)))


# Evaluación

# In[168]:


network.eval()


# In[169]:


X_test.shape


# In[170]:


type(X_test)


# In[171]:


#Convirtiendo de tipo numpy a tensor
X_test = torch.tensor(X_test, dtype=torch.float)
type(X_test)


# In[172]:


predictions = network.forward(X_test)
predictions


# In[173]:


#transformación de la predicciones
predictions = np.array(predictions > 0.5)
predictions


# In[174]:


y_test


# In[175]:


accuracy_score(y_test, predictions)


# In[176]:


cm = confusion_matrix(y_test, predictions)
cm


# In[177]:


import seaborn as sns
sns.heatmap(cm, annot=True)


# Red Neuronal Convolucional

# In[137]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_sample_images


# Conjunto de datos Datos

# In[138]:


#cargando conjuntos de datos
breast= datasets.load_breast_cancer()
#ver lo datos
breast.data


# In[202]:


print(breast.keys())


# In[209]:


print(breast.DESCR)


# In[139]:


NUM_TRAIN = 426
NUM_VAL = 143
NUM_TEST = 143
MINIBATCH_SIZE = 16

transform_breast = T.Compose([T.ToTensor(),T.Normalize([0.491,0.482,0.447],[0.247,0.243,0.261])])
test_loader = DataLoader(X_test, batch_size=MINIBATCH_SIZE,sampler=sampler.SubsetRandomSampler(range(NUM_VAL, len(X_test))))


# Mostrar Imagenes
# 

# In[219]:


plt.show(transform_breast)


# In[ ]:




