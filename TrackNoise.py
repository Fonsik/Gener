#!/usr/bin/env python

from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
import tensorflow as tf
import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=I,I:AnalysisType=Classification')


data = TFile.Open('track.root')
noise = TFile.Open('noise.root')
signal = data.Get('trk')
background = noise.Get('nis')

dataloader = TMVA.DataLoader('dataset')

for branch in signal.GetListOfBranches():
    dataloader.AddVariable(branch.GetName())

dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'SplitMode=Random:NormMode=NumEvents:!V')

# Generate model

# Define model
model = Sequential()
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=784))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=784))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=784))
model.add(Dense(2, activation='softmax'))

# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01), metrics=['accuracy', ])

# Store model to file
model.save('model.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   'H:!V:VarTransform=I,I:FilenameModel=model.h5:NumEpochs=50:BatchSize=32')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
