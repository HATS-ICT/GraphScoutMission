# select which map to use
MAP_LOOKUP = {
    "Std": "_116",
}


PATH_LOOKUP = {
    "file_o": "scout_graph/envs/data/parsed/",
    "file_i": "scout_graph/envs/data/raw/",
}


# assign a large int for no connectivity to replace "null" in graph generation pipeline
INDEX_INVAL = 911


# raw data files for parsing
RAW_DATA_LOOKUP = {
    "f_connectivity": "connectivity_NSWE_116.txt",
    "f_coordinates": "node_lookuptable_116.txt",
    "f_visibility": {0:"visibility_source-stand_target-stand_FOV180.txt",
                    1:"visibility_source-stand_target-prone_FOV180.txt",
                    2:"visibility_source-prone_target-stand_FOV180.txt",
                    3:"visibility_source-prone_target-prone_FOV180.txt"}
}


# lookup table for data types: prefixes of parsed data files for saving and loading
DATA_LOOKUP = {
    "d_connectivity": "graph_move",
    "d_visibility": "graph_view",
    "d_coordinates": "dict_abs_pos",
    "d_mapping": "dict_lookup",
}
