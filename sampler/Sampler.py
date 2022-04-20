import pandas as pd
import numpy as np
import os.path
import glob
import ast
import pickle


class Sampler:
    def __init__(self, source_file, train_dir, val_dir, test_dir):
        self.test_df = None
        self.val_df = None
        self.train_df = None
        self.source_file = source_file
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.node_type_set = set()
        self.label_set = set()

    def setup_data(self):
        df = pd.read_csv(self.source_file, index_col='id', header=0).astype({'problem_id': 'string'})
        problem_ids = df['problem_id'].unique()
        train_dfs = []
        val_dfs = []
        test_dfs = []
        for problem_id in problem_ids:
            samples = df.query(f'problem_id == "{problem_id}"').sample(1000)
            split = np.split(samples, [600, 700, 1000])
            train_dfs.append(split[0])
            val_dfs.append(split[1])
            test_dfs.append(split[2])

        self.train_df = pd.concat(train_dfs, ignore_index=True)
        self.val_df = pd.concat(val_dfs, ignore_index=True)
        self.test_df = pd.concat(test_dfs, ignore_index=True)
        #self.train_df.to_csv(os.path.join(self.train_dir, "train.csv"))
        #self.val_df.to_csv(os.path.join(self.val_dir, "val.csv"))
        #self.test_df.to_csv(os.path.join(self.test_dir, "test.csv"))
        return problem_ids

    def create_trees(self):
        for index, row in self.train_df.iterrows():
            source_filepath = f"C:\\Users\\sebas\\Documents\\Masterarbeit\\Project_CodeNet\\data\\{row['problem_id']}" \
                              f"\\{row['language']}\\{row['submission_id']}.{row['filename_ext']}"
            with open(source_filepath, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=source_filepath)
                json_tree, n_nodes = self.__create_json_tree(tree)
                json_data = {'tree': json_tree, 'label': row['problem_id']}
                json_data_filepath = os.path.join(self.train_dir, str(index) + ".pkl")
                with open(json_data_filepath, 'wb') as out_file:
                    pickle.dump(json_data, out_file)

        for index, row in self.val_df.iterrows():
            source_filepath = f"C:\\Users\\sebas\\Documents\\Masterarbeit\\Project_CodeNet\\data\\{row['problem_id']}" \
                              f"\\{row['language']}\\{row['submission_id']}.{row['filename_ext']}"
            with open(source_filepath, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=source_filepath)
                json_tree, n_nodes = self.__create_json_tree(tree)
                json_data = {'tree': json_tree, 'label': row['problem_id']}
                json_data_filepath = os.path.join(self.val_dir, str(index) + ".pkl")
                with open(json_data_filepath, 'wb') as out_file:
                    pickle.dump(json_data, out_file)

        for index, row in self.test_df.iterrows():
            source_filepath = f"C:\\Users\\sebas\\Documents\\Masterarbeit\\Project_CodeNet\\data\\{row['problem_id']}" \
                              f"\\{row['language']}\\{row['submission_id']}.{row['filename_ext']}"
            with open(source_filepath, encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=source_filepath)
                json_tree, n_nodes = self.__create_json_tree(tree)
                json_data = {'tree': json_tree, 'label': row['problem_id']}
                json_data_filepath = os.path.join(self.test_dir, str(index) + ".pkl")
                with open(json_data_filepath, 'wb') as out_file:
                    pickle.dump(json_data, out_file)

    def __create_json_tree(self, tree):
        n_nodes = 1
        queue = [tree]
        node_type = self.__node_type_name(tree)
        self.node_type_set.add(node_type)
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
                self.node_type_set.add(node_type)
                child_json = {
                    "node": node_type,
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        return root_json, n_nodes

    def __node_type_name(self, node):
        return type(node).__name__

    def get_node_type_set(self):
        return self.node_type_set
