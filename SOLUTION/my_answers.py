import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

#%%
# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    idx = 0 # index of the first element of the window
    while(idx + window_size  < len(series)):
        X.append(series[idx:idx + window_size])
        y.append(series[idx + window_size])
        idx += 1

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

#%% TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

# %% TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique = set()
    for cc in text:
        unique.add(cc)

    # remove as many non-english characters and character sequences as you can 
    nonEnglishChar = []
    for cc in unique:
        if not ((cc >= 'a') & (cc <= 'z')) | (cc in [' ', '!', ',', '.', ':', ';', '?']):
            nonEnglishChar.append(cc)
            
    for cc in nonEnglishChar:
        text = text.replace(cc, ' ')
        
    # shorten any extra dead space created above
    text = text.replace('  ',' ')

    return text

#%% TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    idx = 0 # index of the first element of the window
    while(idx + window_size  < len(text)):
        inputs.append(text[idx:idx + window_size])
        outputs.append(text[idx + window_size])
        idx += step_size

    
    return inputs,outputs
