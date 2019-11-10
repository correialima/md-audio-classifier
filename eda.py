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
    fig, axes = plt.subplots(nrows=2, ncols=5, 
                             sharex= False, sharey = True,
                             figsize=(15,5))
    fig.suptitle('Ondas Originais', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            

def create_set(path):
    df = pd.DataFrame(columns = ['dados','source_file','class'])
    for file in sorted(os.listdir(path)):
        data, fs  = librosa.load(path+file, None)
        for i, ini in zip(range(4),range(0, data.shape[0], fs*intervalo)):
            dados = data[ini:(ini+fs*2)]
            source = file
            character_class = file[i]
            df = df.append({'dados': dados, 'source_file': source, 'class': character_class}, ignore_index = True)
    return df

def main():
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
        fft[c] = (np.fft.rfftfreq(len(sinal),d= 1),abs(np.fft.rfft(sinal)/len(sinal)))
    
    plot_signals(fft)
    plt.show()
    
    
if __name__ == '__main__':
    main()