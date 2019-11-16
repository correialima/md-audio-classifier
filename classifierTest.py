
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

# inputs e constantes
PATH_TREINAMENTO = "./TREINAMENTO/"
PATH_VALIDACAO = "./VALIDACAO/"

def main():
    
    X_train, y_train = create_set(PATH_TREINAMENTO)
    X_val, y_val = create_set(PATH_VALIDACAO)

    X_train = normalize(X_train, axis=0, norm='l2')
    X_val = normalize(X_val, axis=0, norm='l2')

    #teste_random_forest(X_train,y_train,X_val,y_val)
    #teste_kNeighbors(X_train,y_train,X_val,y_val)
    #teste_svc(X_train,y_train,X_val,y_val)
    teste_MLP(X_train,y_train,X_val,y_val)

def teste_MLP(X_train,y_train,X_val,y_val):

    classification_results = []    
    
    activation_functions = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs','sgd','adam']
    learning_rates = ['constant','invscaling','adaptive']
    
    combinations = product(activation_functions,solvers,learning_rates)
    
    
    for af,sv,lr in combinations:
        start = time.time()
        classificador = MLPClassifier(
            solver = sv,
            learning_rate = lr,
            activation = af
        )
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_val)
        end = time.time()
        time_elapsed = end-start
        new_result = (
            af,sv,lr,
            classification_report(y_val, y_pred),
            labeled_confusion_matrix(y_val,y_pred),
            time_elapsed
        )
        classification_results.append(new_result)
    
    print("MLP:")
    for af, sv, lr, cr, cm, t in classification_results:
        
        print(f"Função de ativação: {af}, Solver: {sv}, Learning rate: {lr}")
        print(f"Tempo para classificação: {t}")
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
        print(f"Tempo para classificação: {t}")
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
        print(f"Tempo para classificação: {t}")
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

        classificador = RandomForestClassifier(n_estimators=n)
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_val)
        new_result = (
            n,
            classification_report(y_val, y_pred),
            labeled_confusion_matrix(y_val,y_pred)
            
        )
        classification_results.append(new_result)
    
    print("Random Forest:")
    for n, cr, cm in classification_results:
        
        print(f"Num de estimadores: {n}")
        print(cr)
        print(cm)
        print('___________________________________________________')
    
    
if __name__ == '__main__':
    main()
