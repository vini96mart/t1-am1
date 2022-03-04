import pandas as pd 
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

iris = pd.read_csv('iris.data', header=None)

iris.rename(columns={0:'sepala_comprimento',
                     1:'sepala_largura',
                     2:'petala_comprimento',
                     3:'petala_largura',
                     4:'classe'}, inplace=True)


print(iris.classe.value_counts())

print(iris.info())

colors = iris.classe.replace({'Iris-setosa': 'purple',
                              'Iris-versicolor': 'yellow', 
                              'Iris-virginica': 'green'})


fig, (ax1, ax2)  = plt.subplots(2,1, figsize=(10,10),sharex='all', sharey='all')

sepala = ax1.scatter(iris.sepala_comprimento, iris.sepala_largura, 
                    c=colors, marker='.')
ax1.set_title('Sépalas')
ax1.set_xlabel('Comprimento')
ax1.set_ylabel('Largura')
ax1.grid(True)
ax1.legend(handles=[mpatches.Patch(color='purple', label='Iris-setosa'),
                    mpatches.Patch(color='yellow', label='Iris-versicolor'),
                    mpatches.Patch(color='green', label='Iris-virginica')])

petala = ax2.scatter(iris.petala_comprimento, iris.petala_largura,
                    c=colors, marker='.')
ax2.set_title('Pétalas')
ax2.set_xlabel('Comprimento')
ax2.set_ylabel('Largura')
ax2.grid(True)
ax2.legend(handles=[mpatches.Patch(color='purple', label='Iris-setosa'),
                    mpatches.Patch(color='yellow', label='Iris-versicolor'),
                    mpatches.Patch(color='green', label='Iris-virginica')])
plt.show()

plt.figure(figsize=(10, 6))
subplot = plt.subplot(111)
subplot.plot(iris['sepala_comprimento'], iris['sepala_largura'], 'g^', label='Sépala')
subplot.plot(iris['petala_comprimento'], iris['petala_largura'], 'bs', label='Pétala')
subplot.legend()
plt.title('Tamanho em centímetros')
plt.xlabel('Comprimento')
plt.ylabel('Largura')
plt.grid(True)
plt.show()