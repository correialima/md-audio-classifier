import librosa
import pandas as pd
import os
import sys
'''
parte 1

iteração através dos arquivos da pasta, criando dataframe para análise

'''
train_path = './TREINAMENTO/'
val_path = './VALIDACAO/'

dados = []

for file in sorted(os.listdir(train_path)):
    data, fs  = librosa.load(train_path+file, None)
    duracao_total = data.shape[0]/fs
    intervalo = 2
    dados_p_seg = []
    for i, ini in enumerate(range(0, data.shape[0], fs*intervalo)):     
        if i == 4:
            break
        dados.append([data[ini:(ini+fs*intervalo)],file.split('.')[0],file[i]])

train_set = pd.DataFrame(dados, columns = ['dados','source_file','class'])

print(train_set)