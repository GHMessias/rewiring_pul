import json

def json_writer(path, key, value):

    try:
        with open(path, 'r') as file:
            positive_nodes = json.load(file)

    except FileNotFoundError:
        positive_nodes = dict()

    positive_nodes[key] = value

    with open(path, 'w') as file:
        json.dump(positive_nodes, file, indent = 2)