import os
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

from file_manager import load_graph_files


def plot_out(map_dir):
    fig = plt.figure(frameon=False, figsize=(12, 9), facecolor='none')
    plt.axis('off')
    map_info = load_graph_files(env_path=map_dir, map_lookup="Std")
    col_map = ["gold"] * len(map_info.n_coord)
    nx.draw_networkx(map_info.g_move, map_info.n_coord, node_color=col_map, node_size=150, font_size=6,
                     edge_color='#806C2A', width=0.5, arrows=True)
    plt.savefig("graph_move.png", dpi=200, transparent=True)
    nx.draw_networkx_edges(map_info.g_view, map_info.n_coord, edge_color="grey", width=0.3, arrows=False)
    plt.savefig("graph_view.png", dpi=200, transparent=True)
    plt.close()


def plot_subgraph(map_dir):
    map_info = load_graph_files(env_path=map_dir, map_lookup="Std")
    # issue with the current terrain map setup
    # print("[!!] Undesirable moving directions 86-69:")
    # print(f"Valid actions from node: {86} {map_info.get_Gmove_action_node_dict(86)}")
    # print(f"Valid actions from node: {69} {map_info.get_Gmove_action_node_dict(69)}")

    def filter_edge_N(n1, n2):
        return map_info.g_move[n1][n2]["action"] == 1

    def filter_edge_S(n1, n2):
        return map_info.g_move[n1][n2]["action"] == 2

    def filter_edge_W(n1, n2):
        return map_info.g_move[n1][n2]["action"] == 3

    def filter_edge_E(n1, n2):
        return map_info.g_move[n1][n2]["action"] == 4

    dict_dir = {1: "./graph_scout/envs/data/move_N.png", 2: "./graph_scout/envs/data/move_S.png",
                3: "./graph_scout/envs/data/move_W.png", 4: "./graph_scout/envs/data/move_E.png"}
    dict_filter = {1: filter_edge_N, 2: filter_edge_S, 3: filter_edge_W, 4: filter_edge_E}
    dict_col = {1: "blue", 2: "red", 3: "cyan", 4: "purple"}

    for _dir in dict_dir:
        fig = plt.figure(frameon=False, figsize=(12, 9), facecolor='none')
        plt.axis('off')
        nx.draw_networkx(map_info.g_move, map_info.n_coord,
                         node_color="gold", node_size=50,
                         font_size=0, font_color="gold",
                         edge_color="grey", width=0.5, arrows=False)
        g_view = nx.subgraph_view(map_info.g_move, filter_edge=dict_filter[_dir])
        g_sub = nx.DiGraph(g_view.edges())
        coord_sub = dict(map_info.n_coord)
        list_del_key = []
        for key in coord_sub:
            if key not in g_sub.nodes():
                list_del_key.append(key)
        for key in list_del_key:
            coord_sub.pop(key)
        nx.draw_networkx(g_sub, coord_sub, node_color="gold", node_size=150, font_size=6, font_color=dict_col[_dir],
                         edge_color=dict_col[_dir], width=1.5, arrows=True)
        plt.savefig(dict_dir[_dir], dpi=200, transparent=True)
        plt.close()


if __name__ == "__main__":
    # plot_out("./")
    plot_subgraph("./")
