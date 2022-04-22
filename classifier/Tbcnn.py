import tensorflow as tf

from classifier.ContinuousBinaryTreeConvLayer.ContinuousBinaryTreeConvLayer import ContinuousBinaryTreeConvLayer
from classifier.DynamicMaxPoolingLayer.DynamicMaxPoolingLayer import DynamicMaxPoolingLayer


class Tbcnn(tf.keras.Model):
    def __init__(self, embedding_size, conv_output_size, num_classes):
        super(Tbcnn, self).__init__()
        self.conv_layer = ContinuousBinaryTreeConvLayer(embedding_size, conv_output_size)
        self.pooling_layer = DynamicMaxPoolingLayer()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.argmax = None

    def call(self, inputs):
        x = self.conv_layer([inputs[0], inputs[1]])
        self.argmax, x = self.pooling_layer(x)
        return self.classifier(x)

