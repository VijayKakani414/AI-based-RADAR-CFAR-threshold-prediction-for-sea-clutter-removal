import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


'''
Do not use BatchNormalization .'''

def cnn_small(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Small model architecture
    x = Conv2D(8, 3, activation=tf.nn.leaky_relu)(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(16, 3, activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation=tf.nn.leaky_relu)(x)
    outputs = Dense(num_classes,activation=tf.nn.leaky_relu)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def cnn_medium(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Medium model architecture
    x = Conv2D(16, 3, activation='elu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='elu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, 3, activation='elu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu')(x)
    outputs = Dense(num_classes,activation='elu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def cnn_large(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Large model architecture
    x = Conv2D(32, 3, activation=tf.nn.leaky_relu)(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, 3, activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, 3, activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, 3, activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation=tf.nn.leaky_relu)(x)
    outputs = Dense(num_classes, activation=tf.nn.leaky_relu)(x)

    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    return model
