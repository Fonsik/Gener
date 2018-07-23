
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

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

np.random.seed(2348)

batch_size = 20000
batch_size_NN = 128 #128
epochs = 12    #12

# input image dimensions
img_rows, img_cols = 28, 28  #28, 28

####################################################################################################

# Data parameters
det_width = img_rows  # stupid, to be changed
det_depth = img_cols
det_shape = (det_width, det_depth)

# Scale track slopes up so that slopes and intercepts receive equal weights in the loss function.
slope_scale = det_width/2

# Number of tracks in each event follows Poisson distribution
mean_tracks = 1
max_tracks = 1

#probability of noise and the efficiency<1
prob_noise = 0.05
efficiency = 0.60
# ### Functions for toy data generation




def simulate_straight_track(m, b, det_shape):
    """
    Simulate detector data for one straight track.
    Parameters:
        m: track slope parameter
        b: track first-layer intercept parameter (detector entry point)
        det_shape: tuple of detector shape: (depth, width)
    Returns:
        ndarray of binary detector data for one track.
    """
    x = np.zeros(det_shape)
    idx = np.arange(det_shape[0])
    hits = (idx*m + b).astype(int)
    # implement hit efficiency
    for i in range(det_shape[0]):
        if (np.random.random() > efficiency):
            hits[i]=0
    #print ("HITS = ",hits)
    valid = (hits > 0) & (hits < det_shape[1])
    x[idx[valid], hits[valid]] = 1
    return x

# Generator for single-track events
def gen_tracks(batch_size=batch_size, det_shape=det_shape):
    """Arguments:
         batch_size: number of events to yield for each call
       Yields: batches of training data for use with the keras fit_generator function
    """
    while True:
        # Entry and exit points are randomized
        bs = np.random.random_sample(size=batch_size)*det_width
        b2s = np.random.random_sample(size=batch_size)*det_width
        ms = (b2s-bs)/det_depth*slope_scale # scaled slope
        tracks = np.zeros((batch_size, 1, det_depth, det_width))
        targets = zip(bs, ms)
        for i, (b, m) in enumerate(targets):
            tracks[i,0] = simulate_straight_track(m/slope_scale, b, det_shape)
        targets = np.asarray(targets)
        yield tracks, targets

# Generator for multi-track events.
# Each event contains exactly n_tracks tracks.
# The target track parameters are sorted in increasing order of intercept.
def gen_n_tracks(batch_size=batch_size, det_shape=det_shape, n_tracks=mean_tracks):
    gen_single = gen_tracks(batch_size=n_tracks, det_shape=det_shape)
    while True:
        batch_events = np.zeros((batch_size, 1, det_depth, det_width))
        batch_targets = -np.ones((batch_size, n_tracks, 2))
        for n in range(batch_size):
            tracks,targets = gen_single.next()
            batch_events[n,0] = np.clip( sum( tracks ), 0, 1)
            event_targets = np.asarray(targets)
            batch_targets[n] = event_targets[event_targets[:,0].argsort()] # sort by first column
        yield batch_events, batch_targets

# Generator for multi-track events.
# Each event contains up to max_tracks tracks.
# The target track parameters are sorted in increasing order of intercept.
def gen_multi_tracks(batch_size=batch_size, det_shape=det_shape, mean_tracks=mean_tracks):
    gen_single = gen_tracks(batch_size=max_tracks, det_shape=det_shape)
    while True:
        batch_events = np.zeros((batch_size, 1, det_depth, det_width))
        batch_targets = -np.ones((batch_size, max_tracks, 2))
        for n in range(batch_size):
            num_tracks = min( max_tracks, np.random.poisson(mean_tracks) )
            tracks,targets = gen_single.next()
            batch_events[n,0] = np.clip( sum( tracks[:num_tracks] ), 0, 1)
            event_targets = np.asarray(targets[:num_tracks])
            batch_targets[n,:num_tracks] = event_targets[event_targets[:,0].argsort()] # sort by first column
        yield batch_events, batch_targets


def gen_noise(batch_size=batch_size, det_shape=det_shape, prob_noise=prob_noise):

        batch_events = np.zeros((batch_size, 1, det_depth, det_width))
        for n in range(batch_size):

            for i in range(det_depth):
               for j in range(det_width):
                 if np.random.random()<prob_noise:
                     batch_events[n,0,i,j]=1


        yield batch_events


####################################################################################################





##########################################################################################

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs



    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations




#####################################################################################


# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, y_train = gen_n_tracks().next()
x_test, y_test = gen_n_tracks().next()

y_train = y_train/100
y_test = y_test/100

# add noise
noise_train = gen_noise().next()
noise_test = gen_noise().next()
x_train = x_train+noise_train
x_test  = x_test+noise_test


#print(x_train[0,0])
#print("Train targets ",y_train[0])
#print(x_test[0,0])
#print("Test targets ",y_test[0])


#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     y_train = y_train.reshape(y_train.shape[0], 1, 2)
#     y_test = y_test.reshape(y_test.shape[0], 1, 2)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     y_train = y_train.reshape(y_train.shape[0], 2, 1)
#     y_test = y_test.reshape(y_test.shape[0], 2, 1)
#     input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('int')
x_test = x_test.astype('int')
x_train = np.clip(x_train, 0, 1)
x_test = np.clip(x_test, 0, 1)
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



# model.add(Flatten(input_shape=input_shape))
# #model.add(Reshape(input_shape - (1, ), input_shape=input_shape))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# #model.add(Dense(32, activation='relu'))
# #model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='tanh'))


model.add(Reshape((1, 2),input_shape=(2,)))


# print outputs for each layer
outputs =  [layer.output for layer in model.layers]
print (outputs)
print (model.summary())

model.compile(loss='mean_squared_error', optimizer='Adam')
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])

# Vizualizing model structure

sequential_model_to_ascii_printout(model)


history=model.fit(x_train, y_train,
          batch_size=batch_size_NN,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# make a prediction
#batch_size=2
y_new = model.predict(x_test)

model.save('cnn.h5')
print (model.summary())

# show the inputs and predicted outputs
#for i in range(min(20,len(x_test))):
#	print("True=%s, Predicted=%s" % ( y_test[i],y_new[i]))

# print activations at layer dense_2
#get_activations(model,x_test,layer_name="dense_2")
#print (model.layers("dense_5").output.get_shape())


# list all data in history
print(history.history.keys())
# summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot fitted parameters
y0_true = []
y0_pred = []
for i in range(len(x_test)):
    y0_true.append((y_test[i])[0])
    y0_pred.append((y_new[i])[0])
    #print (y0_true[i], y0_pred[i])
plt.plot(y0_true, y0_pred, "o")
plt.ylabel('predicted')
plt.xlabel('true')
plt.show()

# the histogram of the data
columns_true = zip(*y0_true) #transpose rows to columns
columns_pred = zip(*y0_pred) #transpose rows to columns
#diff = columns_pred-columns_true
diff = tuple(np.subtract(columns_pred,columns_true))

plt.subplot(1, 2, 1)
n, bins, patches = plt.hist(diff[0], 50, normed=1, facecolor='green', alpha=0.75)
# Fit a normal distribution to the data:
mu, std = norm.fit(diff[0])
# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.title("slope: mu = %.2f,  std = %.2f" % (mu, std))
plt.plot(x, p, 'k', linewidth=2)
plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(diff[1], 50, normed=1, facecolor='green', alpha=0.75)
# Fit a normal distribution to the data:
mu, std = norm.fit(diff[1])
# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.title("offset: mu = %.2f,  std = %.2f" % (mu, std))
plt.plot(x, p, 'k', linewidth=2)
plt.show()
