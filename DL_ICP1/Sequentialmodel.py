import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import LabelEncoder
# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Breas Cancer.csv").values
x = dataset[:,2:32]
#print(x)
encoder = LabelEncoder()
#encoder.fit(x)
encoder_y = encoder.fit_transform(dataset[:,1])
# print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:32], encoder_y,
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(8, activation='relu')) # hidden layer
my_first_nn.add(Dense(5, activation='relu')) # hidden layer
#my_first_nn.add(Dense(15, activation='relu')) # hidden layer
#my_first_nn.add(Dense(12, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))