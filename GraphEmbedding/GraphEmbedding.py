import networkx as nx
from karateclub.graph_embedding import GL2Vec
import pydot
import os
import numpy as np


def get_subdirectories(main_dir):
    sub_dir_list = []
    for sub_dir in os.listdir(main_dir):
        sub_dir_path = os.path.join(main_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            sub_dir_list.append(sub_dir_path)
    return sub_dir_list


def get_files(directory):
    target_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".dot"):
            full_path = os.path.join(directory, filename)
            if not os.path.isfile(full_path):
                continue
            target_files.append(full_path)
    return target_files


def get_nx_graph(dot_file_path):
    try:
        graphs = pydot.graph_from_dot_file(dot_file_path)
        converted_graph = nx.nx_pydot.from_pydot(graphs[0])
        graph = nx.classes.graph.Graph()

        # GL2Vec document says: "The procedure assumes that nodes have no string feature present"
        for n in converted_graph.nodes:
            if n == '\\n':
                continue
            graph.add_node(int(n))

        for e in converted_graph.edges:
            graph.add_edge(int(e[0]), int(e[1]))

        return graph
    except:
        # TODO: Log
        print(dot_file_path)


def get_graph_list(dot_files_path):
    graph_list = []
    target = []
    for sub_dir in get_subdirectories(dot_files_path):
        for file_path in get_files(sub_dir):
            graph_list.append(get_nx_graph(file_path))
            target.append(os.path.basename(sub_dir))
            
    target = np.array(target)
    return graph_list, target


def get_embedding(graph_list):
    model = GL2Vec()
    model.fit(graph_list)
    return model.get_embedding()


def main():
    dot_files_path = "D:\\Material\\DOT"
    graph_list, target = get_graph_list(dot_files_path)
    embedding = get_embedding(graph_list)

    np.savetxt('target.txt', target, fmt='%s')
    np.savetxt('vector.txt', embedding, fmt='%.18e')


if __name__ == "__main__":
    main()
