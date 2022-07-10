import tensorflow as tf


class AstNode2Vec(tf.keras.Model):
    def __init__(self, embedding_size, vocab_size):
        super(AstNode2Vec, self).__init__()
        self.w_l = self.add_weight(
            shape=(embedding_size, vocab_size), initializer="random_normal", trainable=True, name="w_l"
        )
        self.w_r = self.add_weight(
            shape=(embedding_size, vocab_size), initializer="random_normal", trainable=True, name="w_r"
        )
        self.target_embedding = tf.keras.layers.Embedding(vocab_size+1,
                                                          embedding_size,
                                                          mask_zero=True,
                                                          name="target_embedding")
        self.b = self.add_weight(shape=(embedding_size,), initializer="zeros", trainable=False)
        self.classifier = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        children_ = inputs
        children = self.target_embedding(children_)
        c_r = self.eta_r(children)
        c_l = self.eta_l(children)
        coef = tf.stack([c_l, c_r], axis=2)
        num_children = tf.shape(children)[1]
        batch_size = tf.shape(children)[0]
        coef_reshape = tf.reshape(coef, (batch_size, num_children, 2))

        cont_binary_node = tf.matmul(children, coef_reshape, transpose_a=True)
        w = tf.stack([self.w_l, self.w_r], axis=0)
        context = tf.tensordot(cont_binary_node, w, [[1, 2], [0, 1]])

        result = context
        return result
        #return self.classifier(result)

    def eta_l(self, children):
        num_children = tf.shape(children)[1]
        children_indices = tf.range(1, num_children + 1, 1)
        eta_l = tf.where(
            tf.equal(num_children, 1),
            tf.constant(1, dtype=tf.float32),
            tf.cast(tf.divide(num_children - children_indices, num_children - 1), dtype=tf.float32)
        )
        batch_size = tf.shape(children)[0]
        eta_l_batch = tf.tile(tf.expand_dims(eta_l, axis=1), [batch_size, 1])
        return eta_l_batch

    def eta_r(self, children):
        num_children = tf.shape(children)[1]
        children_indices = tf.range(1, num_children + 1, 1)
        eta_r = tf.where(
            tf.equal(num_children, 1),
            tf.constant(1, dtype=tf.float32),
            tf.cast(tf.divide(children_indices - 1, num_children - 1), dtype=tf.float32)
        )
        batch_size = tf.shape(children)[0]
        eta_r_batch = tf.tile(tf.expand_dims(eta_r, axis=1), [batch_size, 1])
        return eta_r_batch
