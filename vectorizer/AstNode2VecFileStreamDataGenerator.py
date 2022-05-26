import tensorflow as tf
import numpy as np
import pickle


class AstNode2VecFileStreamDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, node_map,
                 batch_size=32, shuffle=False):
        self.file_paths = file_paths
        self.node_map = node_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = []
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_of_files = [self.file_paths[i] for i in indexes]
        X, Y = self.__data_generation(batch_of_files)
        return X, Y

    def __data_generation(self, files):
        'Generates data containing batch_size samples'
        b_nodes, b_children, b_labels = [], [], []
        trees = []
        for file in files:
            with open(file, 'rb') as f:
                tree = pickle.load(f)
                trees.append(tree)

        for tree in trees:
            nodes, children, label = self.__gen_sample(tree)
            b_nodes.append(nodes)
            b_children.append(children)
            b_labels.append(label)

        b_children = \
            self.__pad_batch(b_children)
        return [np.array(b_nodes), np.array(b_children)], np.array(b_labels)

    def __gen_sample(self, sample):
        children = []
        label = self.__onehot(self.node_map[sample['label']], len(self.node_map))
        target = self.node_map[sample['target']]
        for child in sample['children']:
            children.append(self.__onehot(self.node_map[child], len(self.node_map)))
        return target, children, label

    def __pad_batch(self, children):
        feature_len = len(self.node_map)
        if not children[0]:
            return [[[0] * feature_len]]
        max_nodes = max([len(x) for x in children])
        # pad batches so that every batch has the same number of children
        children = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in children]
        return children

    def __onehot(self, i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]