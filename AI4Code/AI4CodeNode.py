import json


class AI4CodeNode:
    def __init__(self, id, type_rule_name):
        self.__id = id
        self.__type_rule_name = type_rule_name
        self.__children = []

    @staticmethod
    def from_json(json_dict):
        return AI4CodeNode(json_dict['id'], json_dict['type-rule-name'])

    def get_children(self):
        return self.__children

    def get_type_rule_name(self):
        return self.__type_rule_name
