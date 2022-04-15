import tensorflow as tf

from classifier.ContinuousBinaryTreeConvLayer.ContinuousBinaryTreeConvLayer import ContinuousBinaryTreeConvLayer
from classifier.DynamicMaxPoolingLayer.DynamicMaxPoolingLayer import DynamicMaxPoolingLayer


class Tbcnn(tf.keras.Model):
    def __init__(self, embedding_size, conv_output_size, num_classes):
        super(Tbcnn, self).__init__()
        self.conv_layer = ContinuousBinaryTreeConvLayer(embedding_size, conv_output_size)
        self.pooling_layer = DynamicMaxPoolingLayer()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.node_input = tf.keras.Input(shape=(None, embedding_size), dtype=tf.float32)
        self.children_input = tf.keras.Input(shape=(None, None), dtype=tf.int32)

    def call(self, inputs):
        x = self.conv_layer([inputs[0], inputs[1]])
        x = self.pooling_layer(x)
        return self.classifier(x)

