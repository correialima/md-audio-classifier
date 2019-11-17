#encoding: utf-8
'''
Projeto de Classificador de Áudio

Daniel Pereira Cinalli - 11069711
Rafael Correia de Lima - 21004515

'''

# bibliotecas para pré-processamento
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# bibliotecas para classificação
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

# bibliotecas para avaliação e exibição dos resultados
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from util import create_set, labeled_confusion_matrix
from config import PATH_TREINAMENTO, PATH_VALIDACAO


X_train, y_train = create_set(PATH_TREINAMENTO)
X_val, y_val = create_set(PATH_VALIDACAO)

X_train = normalize(X_train, axis=0, norm='l2')
X_val = normalize(X_val, axis=0, norm='l2')

classificador = RandomForestClassifier(n_estimators=1000)
classificador.fit(X_train, y_train)

# classificação final e avaliação dos resultados

y_pred = classificador.predict(X_val)

print(classification_report(y_val, y_pred))

print("Matriz de confusão")
print(labeled_confusion_matrix(y_val,y_pred))





