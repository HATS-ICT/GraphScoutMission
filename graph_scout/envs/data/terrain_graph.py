import networkx as nx
import pickle


class MapInfo:
    # simulation terrain graphs with up to 4-way connected grid-like waypoints and FOV-based visibilities & damage probabilities
    def __init__(self):
        self.g_move = nx.DiGraph(method="get_action")  # {node: index, node_label: height, edge_label: direction}
        self.g_view = nx.MultiDiGraph(method="get_distance")  # {node: index, edge_label: direction & posture & probabilities & distance}
        self.n_table = dict()  # {n_id: (row, col)} relative 2D coordinates
        self.n_coord = dict()  # {n_id: (X, Z)} absolute 3D coordinates for visualization
        self.counter = 0

    def add_node_init_list(self, list_n_id) -> bool:
        # if not self.counter:
        #     return True
        # fast init without sanity checks
        self.g_move.add_nodes_from(list_n_id)
        self.g_view.add_nodes_from(list_n_id)
        self.counter = len(list_n_id)
        return False

    def add_node_Gmove_single(self, n_id, **dict_attrs) -> bool:
        # add node to action graph with attrs
        if n_id in self.n_table:
            return True
        self.g_move.add_node(n_id, dict_attrs)
        return False

    def add_node_Gview_single(self, n_id, **dict_attrs) -> bool:
        # add node to visual graph with attrs
        if n_id in self.n_table:
            return True
        self.g_view.add_node(n_id, dict_attrs)
        return False

    def add_edge_Gmove(self, u_id, v_id, attr_value) -> bool:
        # check node existence first
        if u_id in self.n_table and v_id in self.n_table:
            # attr value: action lookup indexing number
            self.g_move.add_edge(u_id, v_id, action=attr_value)
            return False
        else:
            raise ValueError("[GSMEnv][Graph] Invalid node index.")
        return True

    def add_edge_Gview_FOV(self, u_id, v_id, attr_dir, attr_pos, attr_prob, attr_dist) -> bool:
        # check node existence first
        if u_id in self.n_table and v_id in self.n_table:
            # set the distance attribute to the first edge if there are parallel edges
            if self.g_view.has_edge(u_id, v_id):
                self.g_view.add_edge(u_id, v_id, dir=attr_dir, posture=attr_pos, prob=attr_prob)
            else:
                self.g_view.add_edge(u_id, v_id, dir=attr_dir, posture=attr_pos, prob=attr_prob, dist=attr_dist)
            return False
        return True

    def reset(self):
        # if not (nx.is_frozen(self.g_move) and nx.is_frozen(self.g_view)):
        #     self.g_move.clear()
        #     self.g_view.clear()
        self.g_move = nx.DiGraph(method="get_action")
        self.g_view = nx.MultiDiGraph(method="get_distance")
        self.n_table = dict()
        self.n_coord = dict()
        self.counter = 0

    def set_draw_attrs(self, n_id, coord):
        # store 'pos' absolute coordinates attribute for drawing
        if n_id in self.n_table:
            self.n_coord[n_id] = coord
        else:
            raise ValueError("[GSMEnv][Graph] Invalid node index.")
        return False

    def get_graph_size(self):
        return self.counter

    def get_graph_size_verbose(self):
        return self.counter, len(self.g_move), len(self.g_view), len(self.n_table), len(self.n_coord)

    def get_Gmove_edge_attr(self, u_id, v_id):
        # no edge check for fast accessing
        return self.g_move[u_id][v_id]["action"]

    def get_Gview_edge_attr(self, u_id, v_id, e_id):
        # no valid edge check
        return self.g_view[u_id][v_id][e_id]["dir"], self.g_view[u_id][v_id][e_id]["posture"], self.g_view[u_id][v_id][e_id]["prob"]

    def get_Gview_edge_attr_prob(self, u_id, v_id, e_id):
        # no valid edge check
        return self.g_view[u_id][v_id][e_id]["prob"]

    def get_Gview_edge_attr_dir(self, u_id, v_id):
        return self.g_view[u_id][v_id][0]["dist"]

    def get_Gview_edge_attr_dir(self, u_id, v_id, u_dir):
        # check all parallel edges(u, v), return the distance value (>0) if the looking direction is valid
        dirs = [self.g_view[u_id][v_id][index]['dir'] for index in self.g_view[u_id][v_id]]
        # return the distance value or -1 indicator
        return self.g_view[u_id][v_id][u_dir]["dir"] if (u_dir in dirs) else -1

    def get_all_action_Gmove(self, n_id):
        list_t_id = list(nx.neighbors(self.g_move, n_id))
        # get all valid action tokens from 'ACTION_LOOKUP' table
        return [self.get_Gmove_edge_attr(n_id, t_id) for t_id in list_t_id]

    def get_all_state_Gmove(self, n_id):
        adj_id = list(nx.neighbors(self.g_move, n_id))
        # send the whole 1st order subgraph (current_index, list_of_neighbor_index, list_of_action_nums)
        dict_dir_target = dict()
        for t_id in adj_id:
            dict_dir_target[self.get_Gmove_edge_attr(n_id, t_id)] = t_id
        return dict_dir_target

    def get_draw_attr_3D(self):
        # get node positions and labels for connectivity graph visualization
        label_coord = self.n_coord
        label_height = nx.get_node_attributes(self.g_move, "height")
        return label_coord, label_height

    def get_draw_attr_2D(self):
        return self.n_coord

    def get_draw_attr_Gview(self):
        # get node positions and labels for visibility graph visualization
        g_edge_labels = nx.get_edge_attributes(self.g_view, "dist")
        return g_edge_labels

    def save_graph_pickle(self, f_move, f_view, f_table, f_coord):
        # all data saved in the pickle fashion
        nx.write_gpickle(self.g_move, f_move)
        nx.write_gpickle(self.g_view, f_view)
        with open(f_table, 'wb') as file:
            pickle.dump(self.n_table, file, pickle.HIGHEST_PROTOCOL)
        with open(f_coord, 'wb') as file:
            pickle.dump(self.n_coord, file, pickle.HIGHEST_PROTOCOL)

    def load_graph_pickle(self, f_move, f_view, f_table, f_coord) -> bool:
        self.g_move = nx.read_gpickle(f_move)
        self.g_view = nx.read_gpickle(f_view)
        with open(f_table, 'rb') as file:
            self.n_table = pickle.load(file)
        with open(f_coord, 'rb') as file:
            self.n_coord = pickle.load(file)

        # check length
        n_count = len(self.n_table)
        if n_count == len(self.n_coord) and n_count == len(self.g_move):
            self.counter = n_count
            return False
        else:
            raise KeyError("[GSMEnv][Graph] Fatal error while loading parsed pickle files. Please check raw data and try again.")
            return True


