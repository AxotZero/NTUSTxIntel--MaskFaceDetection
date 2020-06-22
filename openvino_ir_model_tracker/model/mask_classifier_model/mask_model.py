from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout

def get_model(shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape, name='conv3x3_1'))
    model.add(Activation('relu', name='Relu_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2x2_1'))
    
    model.add(Conv2D(32, (3, 1), name='conv3x1_1'))
    model.add(Activation('relu', name='Relu_2'))
    
    model.add(Conv2D(64, (3, 3), name='conv3x3_2'))
    model.add(Activation('relu', name='Relu_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2x2_2'))
    
    model.add(Conv2D(32, (1, 1), name='conv1x1_1'))
    model.add(Activation('relu', name='Relu_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2x2_3'))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model