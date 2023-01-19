import tensorflow as tf

from classifier.layers.ContinuousBinaryTreeConvLayer import ContinuousBinaryTreeConvLayer
from classifier.layers.DynamicMaxPoolingLayer import DynamicMaxPoolingLayer


class Tbcnn(tf.keras.Model):
    def __init__(self, feature_size, num_kernels, num_classes):
        super(Tbcnn, self).__init__()
        self.conv_layer = ContinuousBinaryTreeConvLayer(feature_size, num_kernels)
        self.pooling_layer = DynamicMaxPoolingLayer()
        self.hidden = tf.keras.layers.Dense(num_classes, name="classifier")
        self.softmax = tf.keras.layers.Softmax()
        self._name = "TBCNN"

    def call(self, inputs):
        x = self.conv_layer(inputs)
        _, x = self.pooling_layer(x)
        return self.softmax(self.hidden(x))

