import numpy as np
from keras import backend as K
from keras.models import Model
from keras.engine.input_layer import Input
from keras.layers.core import Activation, Dense
from keras.layers import Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from keras.optimizers import  RMSprop

m_bits = 8
k_bits = 8
c_bits = 8
pad = 'same'
m_train = 2**(16)

ainput0 = Input(shape=(m_bits,)) 
ainput1 = Input(shape=(k_bits,)) 
ainput = concatenate([ainput0, ainput1], axis=1)

adense1 = Dense(units=(m_bits + k_bits))(ainput)
adense1a = Activation('tanh')(adense1)
areshape = Reshape((m_bits + k_bits, 1,))(adense1a)
aconv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(areshape)
aconv1a = Activation('tanh')(aconv1)
aconv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(aconv1a)
aconv2a = Activation('tanh')(aconv2)
aconv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(aconv2a)
aconv3a = Activation('tanh')(aconv3)
aconv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(aconv3a)
aconv4a = Activation('sigmoid')(aconv4)
aoutput = Flatten()(aconv4a)
alice = Model([ainput0, ainput1], aoutput, name='alice')


bob = Model([ainput0, ainput1], aoutput, name='bob')

einput = Input(shape=(c_bits,)) #ciphertext only
edense1 = Dense(units=(c_bits + k_bits))(einput)
edense1a = Activation('tanh')(edense1)
edense2 = Dense(units=(c_bits + k_bits))(edense1a)
edense2a = Activation('tanh')(edense2)
ereshape = Reshape((c_bits + k_bits, 1,))(edense2a)
econv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(ereshape)
econv1a = Activation('tanh')(econv1)
econv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(econv1a)
econv2a = Activation('tanh')(econv2)
econv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(econv2a)
econv3a = Activation('tanh')(econv3)
econv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(econv3a)
econv4a = Activation('sigmoid')(econv4)
eoutput = Flatten()(econv4a)# Eve's attempt at guessing the plaintext
eve = Model(einput, eoutput, name='eve')

aliceout = alice([ainput0, ainput1])
bobout = bob( [aliceout, ainput1] )# bob sees ciphertext AND key
eveout = eve( aliceout )# eve doesn't see the key

eveloss = K.mean(K.sum(K.abs(ainput0 - eveout), axis=-1))
bobloss = K.mean(K.sum(K.abs(ainput0 - bobout), axis=-1))
abeloss = bobloss + K.square(m_bits/2 - eveloss)/( (m_bits//2)**2)

abeoptim = RMSprop(lr=0.005)
eveoptim = RMSprop(lr=0.001)

alicemodel = Model([ainput0, ainput1], aliceout, name='alicemodel')
alicemodel.add_loss(abeloss)
alicemodel.compile(optimizer=abeoptim)

bobmodel = Model([ainput0, ainput1], aliceout, name='bobmodel')
bobmodel.add_loss(abeloss)
bobmodel.compile(optimizer=abeoptim)

# Build and compile the Eve model, used for training Eve net (with Alice frozen)

alice.trainable = False
evemodel = Model([ainput0, ainput1], eveout, name='evemodel')
evemodel.add_loss(eveloss)
evemodel.compile(optimizer=eveoptim)

n_epochs = 20
batch_size = 512
n_batches = m_train // batch_size
abecycles = 1
evecycles = 2
epoch = 0

while epoch < n_epochs:
    for iteration in range(n_batches):
        # Train the A-B+E network
        # alice.trainable = True
        for cycle in range(abecycles):
            # Select a random batch of messages, and a random batch of keys
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            l_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            alice.trainable = True
            alicemodel.train_on_batch([m_batch, k_batch], None)
            al = alicemodel.predict([m_batch, k_batch], None) 
            bobmodel.train_on_batch([al, k_batch], None)
            alice.trainable = False
            evemodel.train_on_batch([al, l_batch], None)
            evemodel.train_on_batch([al, l_batch], None)
    epoch += 1

