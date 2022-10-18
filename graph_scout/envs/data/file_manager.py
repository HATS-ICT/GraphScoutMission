import os
import re

from graph_scout.envs.data.terrain_graph import MapInfo
import graph_scout.envs.data.file_lookup as fc

def load_graph_files(env_path="./", map_lookup="Std"):
    assert check_dir(env_path), "[GSMEnv][Error] Invalid path for loading env data: {}".format(env_path)

    path_data = os.path.join(env_path, fc.PATH_LOOKUP["file_o"])
    assert check_dir(path_data), "[GSMEnv][Error] Can not find data in: {}".format(path_data)

    map_id = fc.MAP_LOOKUP[map_lookup]
    graph_move = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_connectivity"], map_id))
    graph_view = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_visibility"], map_id))
    data_table = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_mapping"], map_id))
    data_coord = find_file_in_dir(path_data, "{}{}.pickle".format(fc.DATA_LOOKUP["d_coordinates"], map_id))

    cur_map = MapInfo()
    cur_map.load_graph_pickle(graph_move, graph_view, data_table, data_coord)

    return cur_map


def generate_graph_files(env_path="./", map_lookup="Std", if_overwrite=True):
    assert check_dir(env_path), "[GSMEnv][Error] Invalid path for graph data files: \'{}\'".format(env_path)
    path_file = os.path.join(env_path, fc.PATH_LOOKUP["file_i"])
    assert check_dir(path_file), "[GSMEnv][Error] Can not find data in: {}".format(path_file)

    path_obj = os.path.join(env_path, fc.PATH_LOOKUP["file_o"])
    if not check_dir(path_obj):
        os.mkdir(path_obj)

    map_id = fc.MAP_LOOKUP[map_lookup]

    # check exists of parsed files [option: overwrite existing files if the flag turns on]
    graph_move, _move = check_file_in_dir(path_obj, "{}{}.pickle".format(fc.DATA_LOOKUP["d_connectivity"], map_id))
    graph_view, _view = check_file_in_dir(path_obj, "{}{}.pickle".format(fc.DATA_LOOKUP["d_visibility"], map_id))
    obj_pos, _pos = check_file_in_dir(path_obj, "{}{}.pickle".format(fc.DATA_LOOKUP["position"], map_id))

    if if_overwrite:
        if True in [_acs, _vis, _pos]:
            print("[GSMEnv][Warning] Overwrite previous saved parsing results in \'{}\'".format(env_path))
        else:
            print("[GSMEnv][Info] Start parsing raw data. Parsed data will be saved in \'{}\'".format(env_path))
    else:
        print("[GSMEnv][Info] Start parsing raw data. Data will *NOT* save to files. [online mode]")

    # check exists of raw data files
    # find node connectivity file
    data_edge_move = find_file_in_dir(path_file, fc.RAW_DATA_LOOKUP["r_connectivity"])
    # find visibility & probablity files
    data_edge_view = find_file_in_dir(path_file, fc.RAW_DATA_LOOKUP["r_visibility"])
    # find node absolute coordinate file
    data_node_coor = find_file_in_dir(path_file, fc.RAW_DATA_LOOKUP["r_coordinates"])

    # IOs in node connectivity file
    file = open(data_edge_move, 'r')
    lines = file.readlines()
    for line in lines:
        nodes = connection_line_parser(line)
        u_name = None
        for idx, node in enumerate(nodes):
            row, col = int(node[0]), int(node[1])
            if row == INDEX_INVAL:  # check placeholder for invalid 'null' actions [skip and continue]
                continue
            node_name = "{}_{}".format(row, col)
            cur_map.add_node_acs(node_name)
            if idx:
                # add edges to all four target nodes
                cur_map.add_node_acs(node_name)
                # index number is the attribute for action lookup NSWE
                cur_map.add_edge_acs(u_name, node_name, idx)
            else:
                # the first node is the source node
                u_name = node_name

    # IOs in node visibility file
    file = open(data_edge_vis, 'r')
    lines = file.readlines()
    for line in lines:
        # get target node and source nodes in four directions
        u_node, v_list_N, v_list_S, v_list_W, v_list_E = visibility_fov_line_parser(line)
        u_name = "{}_{}".format(int(u_node[0][0]), int(u_node[0][1]))
        if u_name in cur_map.n_name:
            node_dict = {1: v_list_N, 2: v_list_S, 3: v_list_W, 4: v_list_E}
            for idx in node_dict:
                for v_node in node_dict[idx]:
                    v_name = "{}_{}".format(int(v_node[0]), int(v_node[1]))
                    if v_name in cur_map.n_name:
                        cur_map.add_edge_vis_fov(u_name, v_name, float(v_node[2]), idx)

    # IOs in node absolute coordinate file
    file = open(data_node_pos, 'r')
    lines = file.readlines()
    for line in lines:
        node, coors = coordinate_line_parser(line)
        node_name = "{}_{}".format(int(node[0][0]), int(node[0][1]))
        if node_name in cur_map.n_name:
            cur_map.n_info[cur_map.n_name[node_name]] = (float(coors[0][0]), float(coors[0][2]))

    # save to file
    if if_overwrite:
        cur_map.save_graph_pickle(graph_move, graph_view, obj_pos)
    return cur_map


def connection_line_parser(s):
    # change 'null' action in the raw data with a placeholder for better iterating and action matching
    s_acts = s.replace("null", "({},{})".format(INDEX_INVAL, INDEX_INVAL))
    s_nodes = re.findall(r"\((\d+),(\d+)\)", s_acts)
    # check if the list contains the source node and its neighbors in all four directions
    assert len(s_nodes) == 5, f"[Parser][Error] Invalid node connections in line: \'{s_nodes}\'"
    return s_nodes


def visibility_line_parser(s):
    s_s, s_t = s.split("\t")
    s_id = re.findall(r"\((\d+),(\d+)\)", s_s)
    d_dist = visual_prob_findall(s_t)
    return s_id, d_dist


def visibility_fov_line_parser(s):
    s_nodes = re.split("\t", s)
    s_id = re.findall(r"\((\d+),(\d+)\)", s_nodes[0])
    # generate lists for all looking directions
    d1_list = visual_prob_findall(s_nodes[1])
    d2_list = visual_prob_findall(s_nodes[2])
    d3_list = visual_prob_findall(s_nodes[3])
    d4_list = visual_prob_findall(s_nodes[4])
    return s_id, d1_list, d2_list, d3_list, d4_list


# get a list of all adjacency nodes with 'dist' and 'prob' strings
def visual_prob_findall(s):
    return re.findall(r"\((\d+),(\d+),(\d+\.?\d*)\)\|(\d)\|\D*\|([0-9.eE\-]+);", s)


# verify probablity values
def visual_prob_check_num(s_prob):
    return re.search(r"(\d)(\.\d*)?([eE][-](\d+))?", s_prob)


# omit body parts check tokens
def visual_prob_elem_parser(s):
    elem_list = re.split(r';',s)
    e_list = []
    for elem in elem_list:
        e_list.append(re.split(r'\|\d\|\D*\|', elem))
    return e_list


def coordinate_line_parser(s):
    idx, coor = s.split("\t")
    n_idx = re.findall(r"\((\d+),(\d+)\)", idx)
    n_coor = re.findall(r"\((\d+\.?\d*),\s(\d+\.?\d*),\s(\d+\.?\d*)\)", coor)
    return n_idx, n_coor


def find_file_in_dir(dir_name: str, file_name: str) -> str:
    path = os.path.join(dir_name, file_name)
    assert os.path.isfile(path), f"[Parser][Error] Can not find file: {path}."
    return path


def check_file_in_dir(dir_name: str, file_name: str) -> (str, bool):
    path = os.path.join(dir_name, file_name)
    return path, os.path.isfile(path)


def check_dir(dir_name: str) -> bool:
    return os.path.exists(dir_name)


# logger for step runs
def save_log_2_file(config, n_step, n_done, agents, prev_obs, actions, obs, rewards, dones=None):
    # ori_stdout = sys.stdout
    _log_path = os.path.join(config["root_path"], config["log_path"])
    if not check_dir(_log_path):
        os.makedirs(_log_path)
    file_path = os.path.join(_log_path, "{}done_{}.txt".format(config["log_prefix"], n_done))
    with open(file_path, 'a+') as f:
        # sys.stdout = f
        _buffer = "Step #{:2d} ".format(n_step)
        for _idx in range(len(agents)):
            _buffer += "| {} HP:{} node:{} dir:{} pos:{} ".format(agents[_idx][0], agents[_idx][3],
                                                                  agents[_idx][1][0], agents[_idx][1][1],
                                                                  agents[_idx][2])
        _buffer += f"| Actions:{actions} | Step rewards:{rewards}"
        if config["log_verbose"]:
            _buffer += f" | Obs_before:{prev_obs} | Obs_after:{obs}"
            if dones is not None:
                _buffer += f" | done:{dones}"
        print(_buffer, file=f)
        # sys.stdout = ori_stdout
    return True


# overview of episode rewards
def log_done_reward(config, n_done, rewards):
    _log_path = os.path.join(config["root_path"], config["log_path"])
    if not check_dir(_log_path):
        os.makedirs(_log_path)
    file_episode = os.path.join(_log_path, config["log_overview"])
    with open(file_episode, 'a+') as f:
        _episode = f"Episode #{n_done:2d} ends with episode_reward:{rewards}"
        print(_episode, file=f)
    file_step = os.path.join(_log_path, f"{config['log_prefix']}done_{n_done}.txt")
    with open(file_step, 'a+') as f:
        _step = f"Episode rewards:{rewards}"
        print(_step, file=f)
    return True


def generate_parsed_data_files():
    # relative path to project root
    _env_path = "./"
    _map_lookup = ["Std"]

    for _map in _map_lookup:
        generate_graph_files(env_path=_env_path, map_lookup=_map, route_lookup=_route_lookup,
                             is_pickle_graph=True, if_overwrite=True)


if __name__ == "__main__":
    generate_parsed_data_files()
