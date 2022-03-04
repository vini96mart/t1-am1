import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split

iris = pd.read_csv('iris.data', header=None)
iris.rename(columns={0:'sepala_altura',
                     1:'sepala_largura',
                     2:'petala_altura',
                     3:'petala_largura',
                     4:'classe'}, inplace=True)

iris_data = iris[['sepala_altura', 'sepala_largura', 'petala_altura', 'petala_largura']]
iris_label = iris['classe']

X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=0.4, 
                                                    random_state=42, 
                                                    stratify=iris_label)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test.value_counts())

clf_gini = tree.DecisionTreeClassifier(criterion='gini', random_state=42) 
clf_gini = clf_gini.fit(X_train, y_train)     
iris_pred_gini = clf_gini.predict(X_test)

clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', random_state=42) 
clf_entropy = clf_entropy.fit(X_train, y_train)     
iris_pred_entropy = clf_entropy.predict(X_test)


acuracia_gini = metrics.accuracy_score(y_test, iris_pred_gini)
acuracia_entropy = metrics.accuracy_score(y_test, iris_pred_entropy)

print('Acurácia com índice Gini: ', acuracia_gini)
print('Acurácia com entropia: ', acuracia_entropy)

graph_gini = tree.export_graphviz(clf_gini, 
                                out_file=None, 
                                feature_names=iris_data.columns.unique(),  
                                class_names=iris_label.unique(),  
                                filled=True,
                                special_characters=True)  
gini = graphviz.Source(graph_gini)  
gini

graph_entropy = tree.export_graphviz(clf_entropy, out_file=None, 
                                feature_names=iris_data.columns.unique(),  
                                class_names=iris_label.unique(),  
                                filled=True,  
                                special_characters=True)  
entropy = graphviz.Source(graph_entropy)  
entropy

# análise de acurácia para diferentes divisões de dado de teste/treino
test_size_array = np.linspace(0.1, 0.9, 9)
print(test_size_array)
for size in test_size_array:
    X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=size, 
                                                    random_state=42, 
                                                    stratify=iris_label)

    clf_gini = tree.DecisionTreeClassifier(criterion='gini', random_state=42) 
    clf_gini = clf_gini.fit(X_train, y_train)     
    iris_pred_gini = clf_gini.predict(X_test)



    clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', random_state=42) 
    clf_entropy = clf_entropy.fit(X_train, y_train)     
    iris_pred_entropy = clf_entropy.predict(X_test)


    acuracia_gini = metrics.accuracy_score(y_test, iris_pred_gini)
    acuracia_entropy = metrics.accuracy_score(y_test, iris_pred_entropy)
    print('-----test_size = ',  size, '----------')
    print('Acurácia com índice Gini: ', acuracia_gini)
    print('Acurácia com entropia: ', acuracia_entropy)

# Impressão do gráfico de acurácias nos diferentes índices    
x = np.linspace(0.1, 0.9, num=9)
y1 = [0.866,0.933,0.934,0.950,0.88,0.900,0.924,0.916,0.925]
print(sum(y1)/9)
print(sum(y2)/9)
y2 = [0.866,0.933,0.891,0.950,0.88,0.900,0.924,0.916,0.925]
plt.figure(figsize=(16, 9))
plt.plot(x,y1, label='Acurácia por Gini')
plt.plot(x,y2, label='Acurácia por entropia')
plt.legend()
plt.title('Acurácia dos resultados da árvore')
plt.xlabel('Razão treino/teste')
plt.ylabel('Acurácia')
plt.show()