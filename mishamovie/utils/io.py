import json


def save_json(obj, filename):
    with open(filename, 'w') as file_stream:
        json.dump(obj, file_stream)


def load_json(filename):
    with open(filename, 'r') as file_stream:
        obj = json.load(file_stream)

    return obj
