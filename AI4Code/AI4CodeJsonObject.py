import json

from AI4Code.AI4CodeTree import AI4CodeTree


class AI4CodeJsonObject:
    def __init__(self, graph):
        self.__graph = graph

    @staticmethod
    def from_json(json_dict):
        graph = json_dict['graph']
        return AI4CodeJsonObject(AI4CodeTree.from_json(graph))

    def get_graph(self):
        return self.__graph
