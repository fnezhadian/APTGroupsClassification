import networkx as nx
from karateclub import Graph2Vec, FeatherGraph
from karateclub.graph_embedding import GL2Vec
import os
import numpy as np


gml_files_path = "D:\\Material\\Current\\Flow\\GML"
dataset_path = "D:\\Material\\Current\\Flow\\Dataset\\GL2Vec"


def get_subdirectories(main_dir):
    sub_dir_list = []
    for sub_dir in os.listdir(main_dir):
        sub_dir_path = os.path.join(main_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            sub_dir_list.append(sub_dir_path)
    return sub_dir_list


def get_gml_files(directory):
    target_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".gml"):
            full_path = os.path.join(directory, filename)
            if not os.path.isfile(full_path):
                continue
            target_files.append(full_path)
    return target_files


def read_graph(gml_file_path):
    retrieved_graph = nx.read_gml(gml_file_path)
    graph = nx.classes.graph.Graph()

    # GL2Vec document says: "The procedure assumes that nodes have no string feature present"
    for n in retrieved_graph.nodes:
        if n == '\\n':
            continue
        graph.add_node(int(n))

    for e in retrieved_graph.edges:
        graph.add_edge(int(e[0]), int(e[1]))

    return graph


def convert_category_to_number(argument):
    switcher = {
        "APT1": 1,
        "APT10": 2,
        "APT19": 3,
        "APT21": 4,
        "APT28": 5,
        "APT29": 6,
        "APT30": 7,
        "DarkHotel": 8,
        "EnergeticBear": 9,
        "EquationGroup": 10,
        "GorgonGroup": 11,
        "Winnti": 12
    }
    return switcher.get(argument, 0)


def get_graph_list():
    graph_list = []
    target = []
    for sub_dir in get_subdirectories(gml_files_path):
        for file_path in get_gml_files(sub_dir):
            print(file_path)
            graph = read_graph(file_path)
            graph_list.append(graph)
            dir_number = convert_category_to_number(os.path.basename(sub_dir))
            target.append(dir_number)

    target = np.array(target)
    return graph_list, target


def get_embedding(graph_list):
    model = GL2Vec()
    # model = Graph2Vec()
    # model = FeatherGraph()
    model.fit(graph_list)
    return model.get_embedding()


def main():
    graph_list, target = get_graph_list()
    embedding = get_embedding(graph_list)

    target_path = os.path.join(dataset_path, "target.txt")
    vector_path = os.path.join(dataset_path, "vector.txt")

    np.savetxt(target_path, target, fmt='%s')
    np.savetxt(vector_path, embedding, fmt='%.18e')


if __name__ == "__main__":
    main()
