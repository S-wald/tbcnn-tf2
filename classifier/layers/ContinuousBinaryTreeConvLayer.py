import tensorflow as tf


class ContinuousBinaryTreeConvLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.w_t = self.add_weight(
            shape=(feature_size, output_size), initializer="random_normal", trainable=True, name="w_t"
        )
        self.w_l = self.add_weight(
            shape=(feature_size, output_size), initializer="random_normal", trainable=True, name="w_l"
        )
        self.w_r = self.add_weight(
            shape=(feature_size, output_size), initializer="random_normal", trainable=True, name="w_r"
        )
        self.b = self.add_weight(shape=(output_size,), initializer="random_normal", trainable=True, name="b")

    def call(self, inputs):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        nodes = inputs[0]
        # children is shape (batch_size x max_tree_size x max_children)
        children = inputs[1]
        children_vectors = self.children_tensor(nodes, children)
        nodes = tf.expand_dims(nodes, axis=2)
        tree_tensor = tf.concat([nodes, children_vectors], axis=2, name='trees')
        c_t = self.eta_t(children)
        c_r = self.eta_r(children, c_t)
        c_l = self.eta_l(children, c_t, c_r)

        coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]

        # reshape for matrix multiplication
        x = batch_size * max_tree_size
        y = max_children + 1
        result = tf.reshape(tree_tensor, (x, y, self.feature_size))
        coef = tf.reshape(coef, (x, y, 3))
        result = tf.matmul(result, coef, transpose_a=True)
        result = tf.reshape(result, (batch_size, max_tree_size, 3, self.feature_size))

        # output is (batch_size, max_tree_size, output_size)
        w = tf.stack([self.w_t, self.w_r, self.w_l], axis=0)
        result = tf.tensordot(result, w, [[2, 3], [0, 1]])

        # output is (batch_size, max_tree_size, output_size)
        return tf.nn.relu(result + self.b, name='conv')

    def children_tensor(self, nodes, children):
        """Build the children tensor from the input nodes and child lookup."""
        max_children = tf.shape(children)[2]
        batch_size = tf.shape(nodes)[0]
        num_nodes = tf.shape(nodes)[1]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, self.feature_size))
        # vector_lookup is (batch_size x num_nodes x feature_size)
        vector_lookup = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children = tf.expand_dims(children, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children = tf.concat([batch_indices, children], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x feature_size)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children, name='children')

    def eta_t(self, children):
        """Compute weight matrix for how much each vector belongs to the 'top'"""
        # children is shape (batch_size x max_tree_size x max_children)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]
        # eta_t is shape (batch_size x max_tree_size x max_children + 1)
        return tf.tile(tf.expand_dims(tf.concat(
            [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
            axis=1), axis=0,
        ), [batch_size, 1, 1], name='coef_t')

    def eta_r(self, children, t_coef):
        """Compute weight matrix for how much each vector belogs to the 'right'"""
        # children is shape (batch_size x max_tree_size x max_children)
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]

        # num_siblings is shape (batch_size x max_tree_size x 1)
        num_siblings = tf.cast(
            tf.math.count_nonzero(children, axis=2, keepdims=True),
            dtype=tf.float32
        )
        # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
        num_siblings = tf.tile(
            num_siblings, [1, 1, max_children + 1], name='num_siblings'
        )
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2, name='mask'
        )

        # child indices for every tree (batch_size x max_tree_size x max_children + 1)
        child_indices = tf.multiply(tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                    axis=0
                ),
                axis=0
            ),
            [batch_size, max_tree_size, 1]
        ), mask, name='child_indices')

        # weights for every tree node in the case that num_siblings = 0
        # shape is (batch_size x max_tree_size x max_children + 1)
        singles = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.fill((batch_size, max_tree_size, 1), 0.5),
             tf.zeros((batch_size, max_tree_size, max_children - 1))],
            axis=2, name='singles')

        # eta_r is shape (batch_size x max_tree_size x max_children + 1)
        return tf.where(
            tf.equal(num_siblings, 1.0),
            # avoid division by 0 when num_siblings == 1
            singles,
            # the normal case where num_siblings != 1
            tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
            name='coef_r'
        )

    def eta_l(self, children, coef_t, coef_r):
        """Compute weight matrix for how much each vector belongs to the 'left'"""
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2,
            name='mask'
        )

        # eta_l is shape (batch_size x max_tree_size x max_children + 1)
        return tf.multiply(
            tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
        )