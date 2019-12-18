from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from parameter import *

# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_Model(training):
  
    inputs = Input(name ='the_input',shape=(300,64,3))

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), name='conv1', kernel_initializer='he_normal')(inputs) 
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  

    inner = Conv2D(128, (3, 3), name='conv2', kernel_initializer='he_normal')(inner)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner) 

    inner = Conv2D(256, (5, 5), name='conv3', kernel_initializer='he_normal')(inner)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    inner = Conv2D(256, (5, 5), name='conv4', kernel_initializer='he_normal')(inner)     
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner) 
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max4')(inner)  

    inner = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='con7')(inner)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    print(inner.shape)

    # To Fully Connected layer
    inner = Reshape(target_shape=((32,512)))(inner)  
    inner = Dense(64, activation='relu', kernel_initializer='he_normal')(inner)

    # LSTM layer
    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal')(inner)  
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_1b)

    lstm1_merged = add([lstm_1, reversed_lstm_1b]) 

    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
    reversed_lstm_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_2b)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(lstm2_merged) 
    
    print(inner.shape)

    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32') 
    input_length = Input(name='input_length', shape=[1], dtype='int64')     
    label_length = Input(name='label_length', shape=[1], dtype='int64')     

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) 


    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)
