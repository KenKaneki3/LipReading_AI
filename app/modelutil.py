import os
import gdown
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

WEIGHTS_PATH = os.path.join('..', 'models', 'checkpoint.weights.h5')
WEIGHTS_URL = 'https://drive.google.com/uc?id=1kHs4JdQWWL3iSmm-CFwMERycRwDB_vap'

def load_model() -> Sequential: 
    # Auto-download weights if not present
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs('../models', exist_ok=True)
        print("Downloading model weights...")
        gdown.download(WEIGHTS_URL, WEIGHTS_PATH, quiet=False)

    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(WEIGHTS_PATH)

    return model