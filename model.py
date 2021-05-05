from keras.models import Sequential
from keras.optimizers import Adam, Adadelta
from keras.layers import Conv1D, Conv2D, ZeroPadding2D, Activation, Input, concatenate, Bidirectional, LSTM, LeakyReLU, GlobalAveragePooling1D, Dropout

from keras.models import Model, model_from_yaml
from keras.datasets import mnist

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform,he_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model,normalize

def build_network(input_shape, embeddingsize, n_filters, kernel_size,strides,n_units):
    network = Sequential()
    network.add(Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='tanh')),
    network.add(LeakyReLU()),
    network.add(MaxPooling1D()),
    network.add(Bidirectional(LSTM(n_units[0], return_sequences=True, activation='tanh'), name = 'lstm_1')),
    network.add(LeakyReLU()),
    network.add(Bidirectional(LSTM(n_units[1], return_sequences=True, activation='tanh'), name = 'lstm_2')),
    network.add(LeakyReLU(name='latent')),
    network.add(Flatten())

    network.add(Dense(int(embeddingsize*2), activation='tanh',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform')
    ),


    network.add(Dense(embeddingsize, activation=None,
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform')
    ),

    #Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

    return network

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sqrt(K.sum(K.square(anchor-positive), axis=-1))
        n_dist = K.sqrt(K.sum(K.square(anchor-negative), axis=-1))

        K.print_tensor(K.mean(n_dist), message='negative:')
        K.print_tensor(K.mean(p_dist), message='positive:')

        K.print_tensor(K.mean(K.maximum(p_dist - n_dist + self.alpha, 0)), message='Tripletloss:')

        return K.mean(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def get_config(self):
      cfg = super().get_config()
      return cfg

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def build_model(input_shape, network, margin=0.2):
    '''
    Define the Keras Model for training
        Input :
            input_shape : shape of input trajectories
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)

    '''
     # Define the tensors for the three input trajectories
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    # Generate the encodings (feature vectors) for the three trajectories
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])

    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)

    # return the model
    return network_train

