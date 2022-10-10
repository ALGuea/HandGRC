import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
from keras.layers import Dense, Dropout, RNN, CuDNNLSTM
from tensorflow.nn import  relu, sigmoid, softmax
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
import time

NAME = f"{360}-SEQ-{60}-PRED-{int(time.time())}"
seed = 7
np.random.seed(seed)


# Loading in the labeled training data sets
# The two classes are "Not a Gesture" and "Probably a Gesture"

M_Data = pd.read_csv('C:\\path\\to\\CSV\\NAG.csv', usecols=[1,2,3,4,5,6])
G_Data = pd.read_csv('C:\\path\\to\\CSV\\PAG.csv', usecols=[1,2,3,4,5,6])

# flattening the dataframe to a list to incorporate temporal component
ml = M_Data.values.flatten().tolist()
gl = G_Data.values.flatten().tolist()

# Splitting data into chunks of 0.5 seconds at 120 fps
m_chunks = np.array([ml[x:x+360] for x in range(0, len(ml), 360)])
g_chunks = np.array([gl[x:x+360] for x in range(0, (len(gl)-240), 360)])

zeros= np.zeros((m_chunks.shape[0],1),dtype=int)
ones= np.ones((g_chunks.shape[0],1),dtype=int)


X = np.concatenate((m_chunks,g_chunks), axis = 0)
y = np.concatenate((zeros,ones), axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 22)

class_names = ['Not A Gesture', 'Probably A Gesture']


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

# Architecture of RNN LSTM, see included figures for detailed breakdown

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X_train, y_train):
	model = Sequential([
		CuDNNLSTM(256, return_sequences=True),
		Dropout(0.2),
		CuDNNLSTM(256),
		Dropout(0.2),
		Dense(256, activation = relu),
		Dense(2, activation = softmax)
		])

	model.compile(optimizer=optimizers.Adam(lr=1e-3, decay=1e-5),
		loss = 'sparse_categorical_crossentropy',
		metrics = ['accuracy']
		)
	tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
	filepath = "LSTM_Final-{epoch:02d}-{val_acc:.3f}"
	checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
	history = model.fit(X_train, y_train, epochs = 20, validation_split=0.05, callbacks=[tensorboard, checkpoint])
	scores = model.evaluate(X_train[test], y_train[test], verbose=0)
	
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)

# Printing visuals of training metrics and results
tensorboard = TensorBoard 	
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

print(X_train.shape)
print(y_test.shape )
print(M_Data.shape)
print(G_Data.shape)
print(M_Data.shape[0] + G_Data.shape[0])
