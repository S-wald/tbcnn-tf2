import tensorflow as tf

from classifier.layers.ContinuousBinaryTreeConvLayer import ContinuousBinaryTreeConvLayer
from classifier.layers.DynamicMaxPoolingLayer import DynamicMaxPoolingLayer


class Tbcnn(tf.keras.Model):
    def __init__(self, feature_size, num_kernels, num_classes):
        super(Tbcnn, self).__init__()
        self.conv_layer = ContinuousBinaryTreeConvLayer(feature_size, num_kernels)
        self.conv_layer2 = ContinuousBinaryTreeConvLayer(num_kernels, num_kernels)
        self.pooling_layer = DynamicMaxPoolingLayer()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.argmax = None

    def call(self, inputs):
        x = self.conv_layer([inputs[0], inputs[1]])
        x = self.conv_layer2([x, inputs[1]])
        self.argmax, x = self.pooling_layer(x)
        return self.classifier(x)

