from audioop import add
import glob
import json
import os.path
import pickle
import random

import numpy as np

from AI4Code.AI4CodeJsonObject import AI4CodeJsonObject


class CrossLangAI4CodeProblemClassificationDataset:
    def __init__(self, location, src_location_cpp, src_location_java, n_train_per_problem, n_val_per_problem, n_test_per_problem):
        self.__location = location
        self.__src_location_cpp = src_location_cpp
        self.__src_location_java = src_location_java
        self.__train_split = n_train_per_problem
        self.__val_split = self.__train_split + n_val_per_problem
        self.__test_split = self.__val_split + n_test_per_problem
        
        self.__train_dir = os.path.join(self.__location, "train")
        self.__val_dir = os.path.join(self.__location, "val")
        self.__test_dir = os.path.join(self.__location, "test")

        self.__vocabulary_cpp = set()
        self.__vocabulary_java = set()
        self.__vocabulary = set()
        self.__vocabulary_map = None

        self.__train_files = []
        self.__val_files = []
        self.__test_files = []

        self.__invalid_samples = set()

    def create(self):
        if any(os.scandir(self.__location)):
            return

        self.__make_dirs()

        print("---- Processing first language ----")
        print(self.__src_location_cpp)
        for problem_id in os.listdir(self.__src_location_cpp):
            print(f"processing {problem_id} ...")
            samples = glob.glob(self.__src_location_cpp + "\\" + problem_id + "\\*.json", recursive=True)
            self.__gen_random_samples(samples, problem_id)
            print(f"... finished {problem_id}")

        self.__vocabulary_cpp = self.__vocabulary.copy()
        
        vocabulary_map = dict([(y, x + 1) for x, y in enumerate(sorted(self.__vocabulary_cpp))])
        with open(os.path.join(self.__location, "vocabulary_map_cpp.pkl"), "wb") as f:
            pickle.dump(vocabulary_map, f)

        print("---- Processing second language ----")
        print(self.__src_location_java)
        self.__vocabulary = set()
        for problem_id in os.listdir(self.__src_location_java):
            print(f"processing {problem_id} ...")
            samples = glob.glob(self.__src_location_java + "\\" + problem_id + "\\*.json", recursive=True)
            self.__gen_random_samples(samples, problem_id)
            print(f"... finished {problem_id}")

        self.__vocabulary_java = self.__vocabulary.copy()
        vocabulary_map = dict([(y, x + 1) for x, y in enumerate(sorted(self.__vocabulary_java))])
        with open(os.path.join(self.__location, "vocabulary_map_java.pkl"), "wb") as f:
            pickle.dump(vocabulary_map, f)

        self.__vocabulary.update(self.__vocabulary_cpp)
        self.__vocabulary_map = dict([(y, x + 1) for x, y in enumerate(sorted(self.__vocabulary))])
        with open(os.path.join(self.__location, "vocabulary_map.pkl"), "wb") as f:
            pickle.dump(self.__vocabulary_map, f)

    def __gen_random_samples(self, samples, label):
        random.shuffle(samples)
        split = np.split(np.array(samples), [self.__train_split, self.__val_split, self.__test_split])
        self.__train_files = split[0]
        self.__val_files = split[1]
        self.__test_files = split[2]
        self.__create_trees(self.__train_files, self.__train_dir, label, add_to_vocab=True)
        self.__create_trees(self.__val_files, self.__val_dir, label)
        self.__create_trees(self.__test_files, self.__test_dir, label)

    def __create_trees(self, files, directory, label, add_to_vocab=False):
        for file in files:
            with open(file, encoding='utf-8') as src_file:
                json_content = json.load(src_file)
                ai4code_obj = AI4CodeJsonObject.from_json(json_content)
                ai4code_tree = ai4code_obj.get_graph()
                tree = ai4code_tree.get_root_node()
                if ai4code_obj.get_graph().get_num_of_nodes() < 2:
                    self.__invalid_samples.add(ai4code_tree.get_src_file())
                    continue
                json_tree = self.__create_json_tree(tree, add_to_vocab)
                json_data = {'tree': json_tree,
                            'label': label}
                submission_id = os.path.basename(file).replace(".json", "")
                json_data_filepath = os.path.join(directory, submission_id + ".pkl")
                with open(json_data_filepath, 'wb') as out_file:
                    pickle.dump(json_data, out_file)

    def __create_json_tree(self, tree, add_to_vocab):
        n_nodes = 1
        queue = [tree]
        node_type = tree.get_type_rule_name()
        node_term = node_type
        if add_to_vocab:
            self.__vocabulary.add(node_term)
        root_json = {
            "node": node_term,
            "children": []
        }
        queue_json = [root_json]
        while queue:
            current_node = queue.pop(0)
            n_nodes += 1
            current_node_json = queue_json.pop(0)

            children = current_node.get_children()
            queue.extend(children)
            for child in children:
                node_type = child.get_type_rule_name()
                node_term = node_type
                if add_to_vocab:
                    self.__vocabulary.add(node_term)
                child_json = {
                    "node": node_term,
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        return root_json

    def __make_dirs(self):
        os.mkdir(self.__train_dir)
        os.mkdir(self.__val_dir)
        os.mkdir(self.__test_dir)

    def get_train_files(self):
        return glob.glob(self.__train_dir + "\\*.pkl")

    def get_val_files(self):
        return glob.glob(self.__val_dir + "\\*.pkl")

    def get_test_files(self):
        return glob.glob(self.__test_dir + "\\*.pkl")

    def get_test_files_cpp(self):
        return glob.glob(self.__test_dir + "\\*.cpp.pkl")

    def get_test_files_java(self):
        return glob.glob(self.__test_dir + "\\*.java.pkl")

    def get_vocabulary_map(self):
        with open(os.path.join(self.__location, "vocabulary_map.pkl"), "rb") as f:
            self.__vocabulary_map = pickle.load(f)
        return self.__vocabulary_map

    def get_labels(self):
        return os.listdir(self.__src_location_cpp)

    def get_invalid_samples(self):
        return self.__invalid_samples
