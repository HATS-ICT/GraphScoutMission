# default local configs

init_setup = {
"INIT_ENV": {
"env_path": './',
"map_id": 'Std',    # 116 waypoints without real sub-nodes
"num_sub_step": 4,  # "num_sub_node": 3,
"num_red": 2,
"num_blue": 2,
"agents_init":{
    "red_0":{"type":"RL", "team_id":0, "health": 100, "direction": 2, "posture": 0,
             "node": 96, "is_lr":True, "is_ob":True},
    "red_1":{"type":"RL", "team_id":0, "health": 100, "direction": 2, "posture": 0,
             "node": 112, "is_lr":True, "is_ob":True},
    "blue_0":{"type":"DT", "team_id":1, "health": 200, "direction": 3, "posture": 0,
              "path": [30, 31, 45, 46, 63, 68, 69, 86]},
    "blue_1":{"type":"DT", "team_id":1, "health": 200, "direction": 4, "posture": 0,
              "path": [30, 31, 45, 46, 47, 48, 56, 72, 73, 74, 75, 84]},
},
"engage_range": {
    0: {"dist": 50, "prob_add": 0.1, "prob_mul": 1.0},
    1: {"dist": 150, "prob_add": 0.05, "prob_mul": 1.0},
},
"max_step": 40,
},

"LOGS":{ # Designed for eval/logger. No local log files should be generated during training process.
"log_on": False, "log_path": "logs/", "log_prefix": "log_",
"log_overview": "reward_episodes.txt",
"log_verbose": False, "log_plot": False, "log_save": False,
},


"INIT_REWARDS":{
},

"INIT_LOCAL":{
"masked_act": True,
"masked_map": False,
"sight_range": -1,  # -1 for unlimited range
"damage_single": 10,
"damage_field": 5,
"field_boundary_node": 75,  # machine gun covered region [0 <= node_list_index < 75]
},

"masked_node_list": [], # option: [1 to 20]
"penalty_unmasked_invalid_action": 0, # -1

}

