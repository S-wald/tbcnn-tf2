import pickle

from AI4Code.AI4CodeJsonObject import AI4CodeJsonObject


class AstNode2VecDataset2:
    def __init__(self, file_paths):
        self.__file_paths = file_paths
        self.__samples = []
        self.__vocabulary = set()

    def create(self):
        trees = []
        for file in self.__file_paths:
            with open(file, 'rb') as f:
                tree = pickle.load(f)
                trees.append(tree)
        for tree in trees:
            self.__samples += self.__create_sample(tree)

    def __create_sample(self, tree):
        samples = []
        queue = [tree["tree"]]
        while queue:
            current_node = queue.pop(0)
            node_type = current_node["node"]
            self.__vocabulary.add(node_type)
            sample_json = {
                "target": node_type,
                "context": []
            }
            children = current_node["children"]
            queue.extend(children)
            for child in children:
                child_node_type = child["node"]
                sample_json["context"].append(child_node_type)
            samples.append(sample_json)
        return samples

    def get_samples(self):
        return self.__samples

    def get_vocabulary(self):
        return self.__vocabulary
