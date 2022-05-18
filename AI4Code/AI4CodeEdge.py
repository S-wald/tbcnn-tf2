import json


class AI4CodeEdge:
    def __init__(self, frm, to):
        self.__from = frm
        self.__to = to

    def get_from(self):
        return self.__from

    def get_to(self):
        return self.__to

    @staticmethod
    def from_json(json_dict):
        between = json_dict['between']
        return AI4CodeEdge(between[0], between[1])
