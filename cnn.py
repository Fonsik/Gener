

from __future__ import print_function

import matplotlib.pyplot as plt
# Package imports
import numpy as np
from scipy.stats import norm

from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Reshape
from keras.models import Sequential
from keras_sequential_ascii import sequential_model_to_ascii_printout

batch_size = 50000
batch_size_NN = 128 #128
epochs = 12    #12

# input image dimensions
img_rows, img_cols = 28, 28  #28, 28


x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')
'''
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

for n in range(min(16,len(x_train))):
  plt.subplot(4, 4, n+1)
  plt.imshow(x_train[n,0])
plt.show()

for i in range(img_rows):
    print (" ")
    for j in range (img_cols):
        print (x_train[0,0,i,j], end=' ')
print (" ")
print ("y_train[0] = ",y_train[0])
print ("y_test[0] = ",y_test[0])

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#print ("Class = ",y_train[0])

'''
model = Sequential()
input_shape=(1,img_rows,img_cols)

model.add(Reshape((img_rows,img_cols,1),input_shape=input_shape))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='tanh'))
model.add(Reshape((1, 2),input_shape=(2,)))

outputs =  [layer.output for layer in model.layers]
print (outputs)
print (model.summary())

model.compile(loss='mean_squared_error', optimizer='Adam')


sequential_model_to_ascii_printout(model)


history=model.fit(x_train, y_train,
          batch_size=batch_size_NN,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score)

model.save('cnn.h5')
print (model.summary())
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

