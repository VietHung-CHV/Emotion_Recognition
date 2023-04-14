from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

def simple_CNN(input_shape, num_classes):
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=128, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=256, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=256, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(36))
    model.add(BatchNormalization())
    model.add(Dropout(.5))
    model.add(Dense(5))
    
    


if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 4
    
    model = simple_CNN((48, 48, 1), num_classes)
    model.summary()