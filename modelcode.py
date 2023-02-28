from tensorflow import keras
from tensorflow.keras import layers

def make_c3_model(num_outputs=3):
    inp = layers.Input(shape=(100,100,1))
    x = layers.experimental.preprocessing.RandomRotation(0.5)(inp)
    x = layers.experimental.preprocessing.RandomFlip()(x)

    x = layers.Conv2D(48, kernel_size=5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Conv2D(64, kernel_size=5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Conv2D(96, kernel_size=5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Conv2D(192, kernel_size=3, padding='same', use_bias=False)(x) # orig = 192
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Conv2D(192, kernel_size=3, padding='same', use_bias=False)(x) # orig = 192
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(320)(x) # original = 320
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outp = layers.Dense(num_outputs, activation='softmax')(x)

    model = keras.Model(inputs=inp, outputs=outp)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model