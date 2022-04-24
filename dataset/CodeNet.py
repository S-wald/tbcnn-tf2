import glob
import os
import pandas as pd

submission_metadata_dtypes = {
    "submission_id": "string",
    "problem_id": "string",
    "user_id": "string",
    "date": "Int64",
    "language": "string",
    "original_language": "string",
    "filename_ext": "string",
    "status": "string",
    "cpu_time": "Int64",  # use Int64 because of NaN values
    "memory": "Int64",
    "code_size": "Int64",
    "accuracy": "string"
}

problem_metadata_dtypes = {
    "id": "string",
    "name": "string",
    "dataset": "string",
    "time_limit": "Int64",
    "memory_limit": "Int64",
    "rating": "string",
    "tags": "string",
    "complexity": "string"
}


class CodeNet:
    def __init__(self, location):
        self.location = location
        self.submission_metadata = None
        self.problem_metadata = None

    def get_submission_metadata(self):
        if self.submission_metadata is not None:
            return self.submission_metadata
        metadata_dfs = []
        path = os.path.join(self.location, "metadata")
        files = glob.glob(path + "/*.csv")
        for file in files:
            if file.endswith("problem_list.csv"):
                continue
            df = pd.read_csv(file,
                             index_col=None,
                             header=0,
                             dtype=submission_metadata_dtypes)
            metadata_dfs.append(df)

        self.submission_metadata = pd.concat(metadata_dfs)
        return self.submission_metadata

    def get_problem_metadata(self):
        if self.problem_metadata is not None:
            return self.problem_metadata
        path = os.path.join(self.location, "metadata/problem_list.csv")
        self.problem_metadata = pd.read_csv(path,
                                            index_col=None,
                                            header=0,
                                            dtype=problem_metadata_dtypes)
        return self.problem_metadata

    def get_src_paths_of_submissions(self, submission_ids):
        rows = self.submission_metadata.query(f'submission_id in {submission_ids}')
        paths = {}
        for index, row in rows.iterrows():
            paths[row['submission_id']] = (os.path.join(*[self.location,
                                                          "data",
                                                          row["problem_id"],
                                                          row["language"],
                                                          f"{row['submission_id']}.{row['filename_ext']}"]))
        return paths

    def get_src_code_of_submission(self, submission_id):
        path = list(self.get_src_paths_of_submissions(list(submission_id)).values())[0]
        with open(path, encoding='utf-8') as f:
            return f.read()
