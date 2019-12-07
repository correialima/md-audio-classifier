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
import random
import sys

# bibliotecas para classificação
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

# bibliotecas para avaliação e exibição dos resultados
from sklearn.ensemble import RandomForestClassifier

from util import create_set, labeled_confusion_matrix, rearrange
from config import PATH_TREINAMENTO, PATH_VALIDACAO, PATH_TESTE, AUDIOS_POR_ARQUIVO, SEED

def main_entrega():
    """Versão de entrega. Junta arquivos da pasta de treinamento e validação e prediz para arquivos da pasta teste"""
    random.seed(SEED)
    np.random.seed(SEED)
    
    X_train, y_train = create_set(PATH_TREINAMENTO)
    X_val, y_val = create_set(PATH_VALIDACAO)
    X_test, y_test = create_set(PATH_TESTE)
    
    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))
    
    X_train = normalize(X_train, axis=0, norm='l2')
    X_test = normalize(X_test, axis=0, norm='l2')
    
    classificador = RandomForestClassifier(n_estimators = 75)
    classificador.fit(X_train, y_train)
    
    # classificação final e avaliação dos resultados
    
    print("Validação")
    y_pred = classificador.predict(X_test)
    
    y_test = rearrange(y_test, AUDIOS_POR_ARQUIVO)
    y_pred = rearrange(y_pred, AUDIOS_POR_ARQUIVO)
    
    print(classification_report(y_test, y_pred))
    #print("Matriz de confusão")
    #print(labeled_confusion_matrix(y_test,y_pred))

def main_validacao():
    """Versão de validação. Treina com arquivos da pasta treinamento e prediz arquivos da pasta validação"""
    random.seed(SEED)
    np.random.seed(SEED)
    
    X_train, y_train = create_set(PATH_TREINAMENTO)
    X_val, y_val = create_set(PATH_VALIDACAO)
    
    X_train = normalize(X_train, axis=0, norm='l2')
    X_val = normalize(X_val, axis=0, norm='l2')
    
    classificador = RandomForestClassifier(n_estimators = 75)
    classificador.fit(X_train, y_train)
    
    # classificação final e avaliação dos resultados
    
    print("Validação")
    y_pred = classificador.predict(X_val)
    
    y_val = rearrange(y_val, AUDIOS_POR_ARQUIVO)
    y_pred = rearrange(y_pred, AUDIOS_POR_ARQUIVO)
    
    print(classification_report(y_val, y_pred))
    #print("Matriz de confusão")
    #print(labeled_confusion_matrix(y_val,y_pred))
    
    '''
    # teste
    print("Teste")
    y_pred = classificador.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    print("Matriz de confusão")
    print(labeled_confusion_matrix(y_test, y_pred))
    '''



if __name__ == '__main__':
    """Por padrão executa versão para entrega"""
    if len(sys.argv) == 1:
        main_entrega()
        quit()
    if sys.argv[1] == "validacao":
        main_validacao()
