
import numpy as np
from itertools import product
import time

# bibliotecas para classificação
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

# bibliotecas para avaliação e exibição dos resultados
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

from util import create_set, labeled_confusion_matrix
from config import PATH_TREINAMENTO, PATH_VALIDACAO

def main():
    
    start = time.time()
    
    X_train, y_train = create_set(PATH_TREINAMENTO)
    X_val, y_val = create_set(PATH_VALIDACAO)

    X_train = normalize(X_train, axis=0, norm='l2')
    X_val = normalize(X_val, axis=0, norm='l2')

    end = time.time()
    
    print(f"Tempo para ler arquivos e extrair features: {end-start} s")

    teste_random_forest(X_train,y_train,X_val,y_val)
    #teste_kNeighbors(X_train,y_train,X_val,y_val)
    #teste_svc(X_train,y_train,X_val,y_val)
    #teste_MLP(X_train,y_train,X_val,y_val)

    

def teste_MLP(X_train,y_train,X_val,y_val):

    classification_results = []    
    
    hidden_layer_sizess = [(100, ), (150, ), (200, ), (100, 20), (100, 50 ), (150, 20), (150, 50), (200, 20), (200, 50)]
    activation_functions = ['tanh'] #'identity', 'logistic', 'tanh', 'relu'
    solvers = ['lbfgs'] #'lbfgs','sgd','adam'
    learning_rates = ['constant','invscaling','adaptive'] #'constant','invscaling','adaptive'
    
    combinations = product(
        hidden_layer_sizess,
        activation_functions,
        solvers,
        learning_rates
        )
    
    
    for hls, af, sv, lr in combinations:
        start = time.time()
        classificador = MLPClassifier(
            hidden_layer_sizes = hls,
            solver = sv,
            learning_rate = lr,
            activation = af
        )
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_val)
        end = time.time()
        time_elapsed = end-start
        new_result = (
            hls, af , sv, lr,
            classification_report(y_val, y_pred),
            labeled_confusion_matrix(y_val,y_pred),
            time_elapsed
        )
        classification_results.append(new_result)
    
    print("MLP:")
    for hls, af, sv, lr, cr, cm, t in classification_results:
        
        print(f"Função de ativação: {af}, Solver: {sv}, Learning rate: {lr}, Hidden_layer: {hls}")
        print(f"Tempo para classificação: {t} s")
        print(cr)
        print(cm)
        print('___________________________________________________')


def teste_svc(X_train,y_train,X_val,y_val):

    classification_results = []    
    
    kernels = ['rbf', 'poly', 'sigmoid']
    gammas = ['auto','scale']
    shapes = ['ovr','ovo']
    
    combinations = product(kernels,gammas,shapes)
    
    
    for k, g, s in combinations:
        start = time.time()
        classificador = SVC(
            kernel = k,
            gamma = g,
            decision_function_shape = s
        )
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_val)
        end = time.time()
        time_elapsed = end-start
        new_result = (
            k, g, s,
            classification_report(y_val, y_pred),
            labeled_confusion_matrix(y_val,y_pred),
            time_elapsed
        )
        classification_results.append(new_result)
    
    print("SVC:")
    for k, g, s, cr, cm, t in classification_results:
        
        print(f"kernel: {k}, gamma: {g}, decision function shape: {s}")
        print(f"Tempo para classificação: {t} s")
        print(cr)
        print(cm)
        print('___________________________________________________')
        
def teste_svc(X_train,y_train,X_val,y_val):

    classification_results = []    
    
    kernels = ['rbf', 'poly', 'sigmoid']
    gammas = ['auto','scale']
    shapes = ['ovr','ovo']
    
    combinations = product(kernels,gammas,shapes)
    
    
    for k, g, s in combinations:
        start = time.time()
        classificador = SVC(
            kernel = k,
            gamma = g,
            decision_function_shape = s
        )
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_val)
        end = time.time()
        time_elapsed = end-start
        new_result = (
            k, g, s,
            classification_report(y_val, y_pred),
            labeled_confusion_matrix(y_val,y_pred),
            time_elapsed
        )
        classification_results.append(new_result)
    
    print("SVC:")
    for k, g, s, cr, cm, t in classification_results:
        
        print(f"kernel: {k}, gamma: {g}, decision function shape: {s}")
        print(f"Tempo para classificação: {t} s")
        print(cr)
        print(cm)
        print('___________________________________________________')

def teste_kNeighbors(X_train,y_train,X_val,y_val):
    
    classification_results = []    
    n_neighbors = [i for i in range(5,7)]
    
    for n in n_neighbors:

        classificador = KNeighborsClassifier(n_neighbors=n)
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_val)
        new_result = (
            n,
            classification_report(y_val, y_pred),
            labeled_confusion_matrix(y_val,y_pred)
            
        )
        classification_results.append(new_result)
    
    print("K neighbors:")
    for n, cr, cm in classification_results:
        
        print(f"K = : {n}")
        print(cr)
        print(cm)
        print('___________________________________________________')



def teste_random_forest(X_train,y_train,X_val,y_val):
    
    classification_results = []    
    n_estimators = [100*i for i in range(5,20)]
    
    for n in n_estimators:
        start = time.time()
        classificador = RandomForestClassifier(n_estimators=n)
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_val)
        end = time.time()
        time_elapsed = end-start
        new_result = (
            n,
            classification_report(y_val, y_pred),
            labeled_confusion_matrix(y_val,y_pred),
            time_elapsed
        )
        classification_results.append(new_result)
    
    print("Random Forest:")
    for n, cr, cm, t in classification_results:
        
        print(f"Num de estimadores: {n}")
        print(f"Tempo para classificação: {t} s")
        print(cr)
        print(cm)
        print('___________________________________________________')
    
    
if __name__ == '__main__':
    main()
