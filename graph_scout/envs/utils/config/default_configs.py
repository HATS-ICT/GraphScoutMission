# default local configs

init_setup = {
"INIT_ENV": {
"env_path": './',
"map_id": 'Std',  # map version: 116 waypoints without real sub-nodes

"max_step": 40,
"num_sub_step": 4,

"num_red": 2,
"num_blue": 2,
"health_red": 100,
"health_blue": 200,
"agents_init":{
    "red_0":{"type":"RL", "team_id":0, "direction": 2, "posture": 0,
             "node": 96, "is_lr":True, "is_ob":True},
    "red_1":{"type":"RL", "team_id":0, "direction": 2, "posture": 0,
             "node": 112, "is_lr":True, "is_ob":True},
    "blue_0":{"type":"DT", "team_id":1, "direction": 1, "posture": 0,
              "path": [30, 31, 45, 46, 63, 68, 69, 86]},
    "blue_1":{"type":"DT", "team_id":1, "direction": 4, "posture": 0,
              "path": [30, 31, 45, 46, 47, 48, 56, 72, 73, 74, 75, 84]},
},

"engage_range": {
    3: {"dist": 50, "prob_add": 0.10, "prob_mul": 1.0},
    2: {"dist": 150, "prob_add": 0.05, "prob_mul": 1.0},
    1: {"dist": 300, "prob_add": 0.01, "prob_mul": 1.0},
},
"engage_token": {
    0: "none-visible"
    1: "far in sight",
    2: "yellow zone",
    3: "red zone",
    4: "overlap",
},
"sight_range": 0,     # 0 for unlimited sight range

"damage_single": 10,  # standard damage
"damage_field": 5,    # machine gun major step damage
"field_boundary_node": 75,  # machine gun covered region [0 <= node_list_index < 75]

"num_hibernate": 5,
"buffer_size": 3,
},

"INIT_REWARDS":{
"step":{
    "rew_step_adv": 2,
    "rew_step_dis": -2,
    "rew_step_slow": 1,
    "rew_step_pass": False,},
"episodic":{
    "rew_ep_health":{},
    "rew_ep_delay":{},
    "rew_ep_alive":{},
    "rew_ep_pass": True,},
},

"INIT_LOG":{ # Designed for eval/logger only. Should not be loaded during training.
"log_on": False, "log_path": "logs/", "log_prefix": "log_",
"log_overview": "reward_episodes.txt",
"log_verbose": False, "log_plot": False, "log_save": False,
},

"INIT_LOCAL":{
"masked_act": False,
"masked_obs": False,
"masked_map": False,
"has_sub_node": False,
},

"LOCAL_TRANS":{
"masked_act": "penalty_invalid",
"masked_obs": "masked_node_list",
"masked_map": "masked_node_list",
"has_sub_node": "num_sub_node",
},

"LOCAL_CONTENT":{
"penalty_invalid": 0, # penalty for unmasked invalid MOVE action
"masked_node_list": [(1, 20),],
"num_sub_node": 3, # num of sub waypoints
},

}

