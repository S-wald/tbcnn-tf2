import tensorflow as tf
import numpy as np
import ast
import pickle

class FileStreamDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, node_map,
                 batch_size=32, shuffle=True,
                 node_embeddings=None):
        self.file_paths = file_paths
        self.labels = labels
        self.n_labels = len(labels)
        self.label_lookup = {label: self.__onehot(i, self.n_labels) for i, label in enumerate(labels)}
        self.node_map = node_map
        self.node_embeddings = node_embeddings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = []
        self.on_epoch_end()
        self.shuffle = shuffle


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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
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

        b_nodes, b_children = \
            self.__pad_batch(b_nodes, b_children)
        return [np.array(b_nodes), np.array(b_children)], np.array(b_labels)

    def __gen_sample(self, tree):
        nodes = []
        children = []
        label = self.label_lookup[tree['label']]
        queue = [(tree['tree'], -1)]
        while queue:
            node, parent_ind = queue.pop(0)
            node_ind = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            if self.node_embeddings is not None:
                nodes.append(self.node_embeddings[self.node_map[node['node']]])
            else:
                nodes.append(self.__onehot(self.node_map[node['node']], len(self.node_map)))

        return nodes, children, label

    def __pad_batch(self, nodes, children):
        if not nodes:
            return [], [], []
        max_nodes = max([len(x) for x in nodes])
        max_children = max([len(x) for x in children])
        feature_len = len(nodes[0][0])
        child_len = max([len(c) for n in children for c in n])

        # pad batches so that every batch has the same number of nodes
        nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
        children = [n + ([[]] * (max_children - len(n))) for n in children]

        # pad every child sample so every node has the same number of children
        children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

        return nodes, children

    def __onehot(self, i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]

