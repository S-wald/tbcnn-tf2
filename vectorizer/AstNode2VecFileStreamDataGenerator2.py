import tensorflow as tf
import numpy as np
import pickle


class AstNode2VecFileStreamDataGenerator2(tf.keras.utils.Sequence):
    def __init__(self, samples, vocabulary,
                 batch_size=1, shuffle=False):
        self.samples = samples 
        self.node_map = dict([(y, x+1) for x, y in enumerate(sorted(vocabulary))])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = []
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_of_samples = [self.samples[i] for i in indexes]
        X, Y = self.__data_generation(batch_of_samples)
        return X, Y

    def __data_generation(self, samples):
        'Generates data containing batch_size samples'
        b_context, b_target = [], []
        for s in samples:
            context, target = self.__gen_sample(s)
            b_context.append(context)
            b_target.append(target)

        b_context = \
            self.__pad_batch(b_context)
        return np.array(b_context), np.array(b_target)

    def __gen_sample(self, sample):
        context = []
        target = self.__onehot(self.node_map[sample['target']], len(self.node_map)+1) 
        # +1 because vocab is one-based as 0 is padding
        for c in sample['context']:
            context.append(self.node_map[c]) # use index for later embedding lookup
        if len(context) == 0:
            context.append(0)
        return context, target

    def __pad_batch(self, children):
        max_nodes = max([len(x) for x in children])
        # pad batches so that every batch has the same number of children
        children = [n + ([0] * (max_nodes - len(n))) for n in children]
        return children

    def __onehot(self, i, total):
        assert i < total
        return [1.0 if j == i else 0.0 for j in range(total)]

    def get_node_map(self):
        return self.node_map