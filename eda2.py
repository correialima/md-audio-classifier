import pandas as pd
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

train_path = './TREINAMENTO/'
val_path = './VALIDACAO/'
intervalo = 2
rate = 44100

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(15,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(15,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(15,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(15,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d = 1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)

def create_set(path):
    df = pd.DataFrame(columns = ['dados','source_file','class'])
    for file in sorted(os.listdir(path)):
        data, fs  = librosa.load(path+file, None)
        for i, ini in zip(range(4),range(0, data.shape[0], fs*intervalo)):
            dados = data[ini:(ini+fs*2)] / max(data[ini:(ini+fs*2)])
            source = file
            character_class = file[i]
            df = df.append({'dados': dados, 'source_file': source, 'class': character_class}, ignore_index = True)
    return df

train_set = create_set(train_path)    
#val_set = create_set(val_path)
#print(val_set['dados'])

classes = list(np.unique(train_set['class']))

signals = {}
fft = {}
fbank = {}
mfccs = {}   

    
for c in classes:
    sinal = train_set[train_set['class'] == c].iloc[0,0]
    signals[c] = sinal
    fft[c] = calc_fft(sinal, rate)
    
    
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()