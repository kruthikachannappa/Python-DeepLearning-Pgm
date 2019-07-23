from keras.layers import Input, Dense, regularizers
from keras.models import Model

# use Matplotlib
import matplotlib.pyplot as plt
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
hidden_size = 128
# this is our input placeholder
input_img = Input(shape=(784,))
hidden_1 = Dense(hidden_size, activation='relu')(input_img)
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(hidden_1)



# "decoded" is the lossy reconstruction of the input
hidden_2 = Dense(hidden_size, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(hidden_2)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencode=autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                verbose=2,
                shuffle=True,
                validation_data=(x_test, x_test))


N=5
plt.figure()
plt.plot(np.arange(0, N), autoencode.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), autoencode.history["val_loss"], label="test_loss")
plt.plot(np.arange(0, N), autoencode.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), autoencode.history["val_acc"], label="test_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.show()