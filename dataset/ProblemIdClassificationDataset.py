import ast
import glob
import os
import pandas as pd
import numpy as np
import pickle


class ProblemIdClassificationDataset:
    def __init__(self, location, source_dataset, codenet, n_train_per_problem, n_val_per_problem, n_test_per_problem):
        self.__location = location
        self.__dataset = source_dataset
        self.__codenet = codenet
        self.__train_df = None
        self.__val_df = None
        self.__test_df = None
        self.__train_dir = os.path.join(self.__location, "train")
        self.__val_dir = os.path.join(self.__location, "val")
        self.__test_dir = os.path.join(self.__location, "test")
        self.__n_samples_per_problem = n_train_per_problem + n_val_per_problem + n_test_per_problem
        self.__train_split = n_train_per_problem
        self.__val_split = self.__train_split + n_val_per_problem
        self.__test_split = self.__val_split + n_test_per_problem
        self.__labels = None
        self.__node_types = set()
        self.__node_map = None

    def create_or_load(self):
        if any(os.scandir(self.__location)):
            self.__load()
        else:
            self.__create()

    def __load(self):
        with open(os.path.join(self.__location, "dataset.pkl"), "rb") as f:
            self.__dataset = pickle.load(f)

        with open(os.path.join(self.__location, "train_df.pkl"), "rb") as f:
            self.__train_df = pickle.load(f)

        with open(os.path.join(self.__location, "val_df.pkl"), "rb") as f:
            self.__val_df = pickle.load(f)

        with open(os.path.join(self.__location, "test_df.pkl"), "rb") as f:
            self.__test_df = pickle.load(f)

        with open(os.path.join(self.__location, "labels.pkl"), "rb") as f:
            self.__labels = pickle.load(f)

        with open(os.path.join(self.__location, "node_map.pkl"), "rb") as f:
            self.__node_map = pickle.load(f)

    def __create(self):
        self.__make_dirs()

        with open(os.path.join(self.__location, "dataset.pkl"), "wb") as f:
            pickle.dump(self.__dataset, f)

        self.__labels = self.__dataset['problem_id'].unique()
        train_dfs = []
        val_dfs = []
        test_dfs = []
        for problem_id in self.__labels:
            samples = self.__dataset.query(f'problem_id == "{problem_id}"').sample(self.__n_samples_per_problem)
            split = np.split(samples, [self.__train_split, self.__val_split, self.__test_split])
            train_dfs.append(split[0])
            val_dfs.append(split[1])
            test_dfs.append(split[2])

        self.__train_df = pd.concat(train_dfs, ignore_index=True)
        self.__val_df = pd.concat(val_dfs, ignore_index=True)
        self.__test_df = pd.concat(test_dfs, ignore_index=True)

        with open(os.path.join(self.__location, "train_df.pkl"), "wb") as f:
            pickle.dump(self.__train_df, f)

        with open(os.path.join(self.__location, "val_df.pkl"), "wb") as f:
            pickle.dump(self.__val_df, f)

        with open(os.path.join(self.__location, "test_df.pkl"), "wb") as f:
            pickle.dump(self.__test_df, f)

        with open(os.path.join(self.__location, "labels.pkl"), "wb") as f:
            pickle.dump(self.__labels, f)

        self.__create_trees(self.__train_df, self.__train_dir)
        self.__create_trees(self.__val_df, self.__val_dir)
        self.__create_trees(self.__test_df, self.__test_dir)

        self.__node_map = dict([(y, x + 1) for x, y in enumerate(sorted(self.__node_types))])
        with open(os.path.join(self.__location, "node_map.pkl"), "wb") as f:
            pickle.dump(self.__node_map, f)

    def get_train_files(self):
        return glob.glob(self.__train_dir + "\\*.pkl")

    def get_val_files(self):
        return glob.glob(self.__val_dir + "\\*.pkl")

    def get_test_files(self):
        return glob.glob(self.__test_dir + "\\*.pkl")

    def get_labels(self):
        return self.__labels

    def get_node_map(self):
        return self.__node_map

    def get_test_df(self):
        return self.__test_df

    def __create_trees(self, df, directory):
        submission_ids = df["submission_id"].values
        paths = self.__codenet.get_src_paths_of_submissions(list(submission_ids))
        for index, row in df.iterrows():
            source_filepath = paths[row['submission_id']]
            with open(source_filepath, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=source_filepath)
                json_tree, n_nodes = self.__create_json_tree(tree)
                json_data = {'tree': json_tree, 'label': row['problem_id']}
                json_data_filepath = os.path.join(directory, str(index) + ".pkl")
                with open(json_data_filepath, 'wb') as out_file:
                    pickle.dump(json_data, out_file)

    def __create_json_tree(self, tree):
        n_nodes = 1
        queue = [tree]
        node_type = self.__node_type_name(tree)
        self.__node_types.add(node_type)
        root_json = {
            "node": node_type,
            "children": []
        }
        queue_json = [root_json]
        while queue:
            current_node = queue.pop(0)
            n_nodes += 1
            current_node_json = queue_json.pop(0)

            children = list(ast.iter_child_nodes(current_node))
            queue.extend(children)
            for child in children:
                node_type = self.__node_type_name(child)
                self.__node_types.add(node_type)
                child_json = {
                    "node": node_type,
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        return root_json, n_nodes

    def __node_type_name(self, node):
        return type(node).__name__

    def __make_dirs(self):
        os.mkdir(self.__train_dir)
        os.mkdir(self.__val_dir)
        os.mkdir(self.__test_dir)
