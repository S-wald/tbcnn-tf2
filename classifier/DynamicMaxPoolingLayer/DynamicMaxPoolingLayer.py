import tensorflow as tf


class DynamicMaxPoolingLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=1)
