import networkx as nx
import pydot
import os


dot_files_path = "D:\\Material\\Current\\DOT"
gml_files_path = "D:\\Material\\Current\\GML"


def get_subdirectories(main_dir):
    sub_dir_list = []
    for sub_dir in os.listdir(main_dir):
        sub_dir_path = os.path.join(main_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            sub_dir_list.append(sub_dir_path)
    return sub_dir_list


def get_dot_files(directory):
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
        graphs = pydot.graph_from_dot_file(dot_file_path, encoding="UTF-8")
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


def save_graph(graph, dot_file_path):
    file_name = os.path.basename(dot_file_path)
    pure_file_name = os.path.splitext(file_name)[0]
    dot_file_location = os.path.dirname(os.path.abspath(dot_file_path))
    folder_name = os.path.basename(os.path.normpath(dot_file_location))
    gml_file_location = os.path.join(gml_files_path, folder_name)

    if not os.path.exists(gml_file_location):
        os.makedirs(gml_file_location)

    path = os.path.join(gml_file_location, "{}.gml".format(pure_file_name))
    nx.write_gml(graph, path)


def main():
    for sub_dir in get_subdirectories(dot_files_path):
        for file_path in get_dot_files(sub_dir):
            graph = get_nx_graph(file_path)
            save_graph(graph, file_path)


if __name__ == "__main__":
    main()
