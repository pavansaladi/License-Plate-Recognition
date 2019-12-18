from keras import backend as K
from keras.optimizers import Adadelta,Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *
import os
from keras.callbacks import History
import matplotlib.pyplot as plt
history = History()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# # Model description and training

model = get_Model(training=True)

model.summary()

try:
    model.load_weights('LSTM+BN4--50--0.056.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass


input_shape = (img_w, img_h ) 

tiger_train = TextImageGenerator( batch_size, input_shape)

tiger_val = TextImageGenerator( batch_size, input_shape)

ada = Adadelta()  

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN7--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

# captures output of softmax so we can decode the output during visualization
seqModel = model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(num_train_images / batch_size),
                    epochs=15,
                    callbacks=[checkpoint],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(num_validation_images/ val_batch_size))

train_loss = seqModel.history['loss']
val_loss   = seqModel.history['val_loss']

ep  =range(1,16)

plt.figure()
plt.plot(ep, train_loss,color='blue')
plt.xlabel('Epoch')
plt.ylabel('train_loss')
plt.title('Train_loss vs Epoch')
plt.savefig("train_loss.png")

plt.figure()
plt.plot(ep, val_loss,color = 'red')
plt.xlabel('Epoch')
plt.ylabel('val_loss')
plt.title('Val_loss vs Epoch')
plt.savefig("val_loss.png")

