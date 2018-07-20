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
                       '!V:!Silent:Color:DrawProgressBar:Transformations=I,G:AnalysisType=Classification')


dim=28*28
data = TFile.Open('track.root')
noise = TFile.Open('noise.root')
signal = data.Get('trk')
background = noise.Get('nis')

dataloader = TMVA.DataLoader('dataset')


for branch in signal.GetListOfBranches():
    name = branch.GetName()
    if (name != 'a' and name!='b'):
        dataloader.AddVariable(name)


dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'SplitMode=Random:NormMode=NumEvents:!V')

# Generate model

# Define model
model = Sequential()
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=dim))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=dim))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=dim))
model.add(Dense(2, activation='softmax'))

# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01), metrics=['accuracy', ])

model.save('model.h5')
model.summary()

#---------------------------------------------------------
#---------------------------------------------------------

output2 = TFile.Open('TMVA2.root', 'RECREATE')
factory2 = TMVA.Factory('TMVARegression', output2,
        '!V:!Silent:Color:DrawProgressBar:Transformations=I,G:AnalysisType=Regression')

dataloader2 = TMVA.DataLoader('dataset')

for branch in signal.GetListOfBranches():
    name = branch.GetName()
    if (name != 'a' and name!='b'):
        dataloader2.AddVariable(name)

dataloader2.AddTarget('a')
dataloader2.AddTarget('b')

dataloader2.AddRegressionTree(signal, 1.0)
dataloader2.PrepareTrainingAndTestTree(TCut(''),
        'SplitMode=Random:NormMode=NumEvents:!V')


model2 = Sequential()
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=dim))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=dim))
model2.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=dim))
model2.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=dim))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=dim))
model2.add(Dense(2, activation='linear'))


model2.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))



model2.save('model2.h5')
model2.summary()

factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   'H:!V:VarTransform=I,G:FilenameModel=model.h5:NumEpochs=50:BatchSize=32')


factory2.BookMethod(dataloader2, TMVA.Types.kPyKeras, 'PyKeras2',
        'H:!V:VarTransform=I,G:FilenameModel=model2.h5:NumEpochs=50:BatchSize=32')


# Run training, test and evaluation
factory.TrainAllMethods()
factory2.TrainAllMethods()

factory.TestAllMethods()
factory2.TestAllMethods()

factory.EvaluateAllMethods()
factory2.EvaluateAllMethods()


