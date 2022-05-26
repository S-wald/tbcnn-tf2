import glob
import json
import os.path
import pickle
import random

from AI4Code.AI4CodeJsonObject import AI4CodeJsonObject


class AstNode2VecDataset:
    def __init__(self, location, src_location, n_samples):
        self.__location = location
        self.__src_location = src_location
        self.__vocabulary = set()
        self.__vocabulary.add("0") # padding symbol
        self.__vocabulary_map = None
        self.__submission_id_problem_id_map = {}
        self.__data_dir = os.path.join(self.__location, "data")
        self.__n_samples = n_samples
        self.__sample_count = 0
        self.__invalid_samples = set()
        self.__json_parse_errors = {}

    def create(self):
        if any(os.scandir(self.__location)):
            return

        self.__make_dirs()

        samples = glob.glob(self.__src_location + "\\**\\*.json", recursive=True)
        random.shuffle(samples)
        self.__gen_dataset(samples)

        self.__vocabulary_map = dict([(y, x+1) for x, y in enumerate(sorted(self.__vocabulary))])
        with open(os.path.join(self.__location, "vocabulary_map.pkl"), "wb") as f:
            pickle.dump(self.__vocabulary_map, f)

    def __gen_dataset(self, files):
        for file in files:
            if self.__sample_count > self.__n_samples:
                return
            with open(file, encoding='utf-8') as src_file:
                try:
                    json_content = json.load(src_file)
                    ai4code_obj = AI4CodeJsonObject.from_json(json_content)
                    ai4code_tree = ai4code_obj.get_graph()
                    tree = ai4code_tree.get_root_node()
                    if ai4code_obj.get_graph().get_num_of_nodes() < 2:
                        self.__invalid_samples.add(ai4code_tree.get_src_file())
                        continue
                    id = 1
                    for sample in self.__create_sample(tree):
                        submission_id = os.path.basename(file).replace(".json", "")
                        json_data_filepath = os.path.join(self.__data_dir, submission_id + "_" + str(id) + ".pkl")
                        with open(json_data_filepath, 'wb') as out_file:
                            pickle.dump(sample, out_file)
                        self.__sample_count += 1
                        id += 1
                except Exception as e:
                    self.__json_parse_errors[file] = str(e)

    def __create_sample(self, tree):
        queue = [tree]
        while queue:
            current_node = queue.pop(0)
            node_type = current_node.get_type_rule_name()
            node_label = current_node.get_label()
            node_term = node_type + node_label
            self.__vocabulary.add(node_term)
            sample_json = {
                "target": node_term,
                "children": [],
                "label": node_term
            }
            children = current_node.get_children()
            queue.extend(children)
            for child in children:
                node_type = child.get_type_rule_name()
                node_label = child.get_label()
                node_term = node_type + node_label
                sample_json["children"].append(node_term)
            yield sample_json

    def __make_dirs(self):
        os.mkdir(self.__data_dir)

    def get_data_files(self):
        return glob.glob(self.__data_dir + "\\*.pkl")

    def get_vocabulary_map(self):
        with open(os.path.join(self.__location, "vocabulary_map.pkl"), "rb") as f:
            self.__vocabulary_map = pickle.load(f)
        return self.__vocabulary_map

    def get_invalid_samples(self):
        return self.__invalid_samples

    def get_json_parse_errors(self):
        return self.__json_parse_errors
