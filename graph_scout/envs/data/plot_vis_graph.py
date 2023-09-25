import os
import re
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

from file_manager import generate_graph_files


def plot_vis(map_dir):
    
    map_info = generate_graph_files(env_path=map_dir, map_lookup="Std", if_overwrite=False)
    

    g_vis = nx.create_empty_copy(map_info.g_move, with_data=True)
    vis_coord = dict(map_info.n_coord)
    vis_table = dict(map_info.n_table)
    for index in range(1, 20):
    	g_vis.remove_node(index)
    	del vis_coord[index]
    	del vis_table[index]
    vis_node_list = list(vis_table.keys())
    vis_index_list = list(vis_table.values())
    # print(g_vis)
    # print(vis_coord, vis_table)

    filename = "./graph_scout/envs/data/vis_graph_edges.txt"
    with open(filename, 'r', encoding='utf-16') as file:
    	lines = file.readlines()
    	for line in lines:
    		node_s_str, node_t_str, num_all, num_vis = line.split("\t")
    		# print(node_s_str)
    		(node_s_x, node_s_z) = re.findall(r"\((\d+),(\d+)\)", node_s_str)[0]
    		node_s = (int(node_s_x), int(node_s_z))
    		(node_t_x, node_t_z) = re.findall(r"\((\d+),(\d+)\)", node_t_str)[0]
    		node_t = (int(node_t_x), int(node_t_z))
    		index_s = vis_node_list[vis_index_list.index(node_s)]
    		index_t = vis_node_list[vis_index_list.index(node_t)]
    		g_vis.add_edge(index_s, index_t, all=int(num_all), vis=int(num_vis))
    fig = plt.figure(frameon=False, facecolor='none')
    plt.axis('off')
    col_map = ["gold"] * len(map_info.n_coord)

    nx.draw_networkx(g_vis, vis_coord, node_color="gold", node_size=150, font_size=6, font_color='black', edge_color='grey', width=0.3, arrows=False)

    plt.savefig("./graph_scout/envs/data/vis_graph_edges.png", dpi=300, transparent=True)
    plt.close()

if __name__ == "__main__":
    plot_vis("./")