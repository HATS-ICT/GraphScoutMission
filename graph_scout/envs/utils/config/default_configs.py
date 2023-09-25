# default local configs

init_setup = {
    "INIT_ENV": {
        "env_path": './',
        "map_id": 'Std',  # map version: 116 waypoints without real sub-nodes

        "max_step": 50,
        "num_sub_step": 4,

        "num_red": 2,
        "num_blue": 2,
        "health_red": 150,
        "health_blue": 200,
        "agents_init": {
            "R_0": {"type": "RL", "team_id": 0, "direction": 2, "posture": 0,
                    "node": 96, "is_lr": True, "is_ob": True},
            "R_1": {"type": "RL", "team_id": 0, "direction": 2, "posture": 0,
                    "node": 112, "is_lr": True, "is_ob": True},
            "B_0": {"type": "DT", "team_id": 1, "direction": 1, "posture": 0,
                    "path": [30, 31, 45, 46, 63, 68, 69, 86]},
            "B_1": {"type": "DT", "team_id": 1, "direction": 4, "posture": 0,
                    "path": [30, 31, 45, 46, 47, 48, 56, 72, 73, 74, 75, 84]},
        },

        "engage_range": {
            3: {"dist": 50, "prob_add": 0.10, "prob_mul": 1.0},
            2: {"dist": 150, "prob_add": 0.05, "prob_mul": 1.0},
            1: {"dist": 300, "prob_add": 0.01, "prob_mul": 1.0},
        },
        "engage_token": {
            0: "none-visible",
            1: "far in sight",
            2: "yellow zone",
            3: "red zone",
            4: "overlap",
        },
        "sight_range": 0,  # 0 for unlimited sight range

        "damage_single": 10,  # standard damage
        "damage_field": 0,  # machine gun major step damage
        "field_boundary_node": 75,  # machine gun covered region [0 <= node_list_index < 75]

        "num_hibernate": 4,  # heuristic agents: initial stay period
        "buffer_timeout": 4,  # heuristic agents: maintain current target for a while (n_major_step)
        "behavior_lookup": {"val": [5, 3, 2, 0], "bar": [0.25, 0.4, 0.5, 1.01]},  # heuristic: health % -> action branch

        "log_on": False
    },

    "INIT_REWARD": {
        "step": {
            "rew_step_adv": 2,
            "rew_step_dis": -2,
            "rew_step_slow": 1,
            "rew_step_on": True},
        "episode": {
            "rew_ep_health": {"type": "table", "num": [30, 20, 10, 5, 1], "bar": [0.05, 0.25, 0.5, 0.75, 1.]},
            "rew_ep_delay": {"type": "steps", "inc": [0, 0.5, 1, 2, 3, 5, 5], "step": [0, 16, 24, 32, 40, 48, 255], "max": 70},
            "rew_ep_bonus": {"type": "thres", "value": 10, "bar": 33},  # 4 + 12 * 2.5
            "rew_ep_on": True},
    },

    "INIT_LOG": {  # Designed for eval/logger only. Should not be loaded during training.
        "log_path": "logs/", "log_prefix": "log_",
        "log_overview": "reward_episodes.txt",
        "log_verbose": False, "log_plot": False, "log_save": False,
    },

    "INIT_LOCAL": {
        "masked_act": True,
        "masked_obs": False,
        "masked_map": False,
        "imbalance_prob": True,
        "has_sub_node": False,
    },

    "LOCAL_TRANS": {
        "masked_act": "penalty_invalid",
        "masked_obs": "masked_node_list",
        "masked_map": "masked_node_list",
        "imbalance_prob": "imbalance_pairs",
        "has_sub_node": "num_sub_node",
    },

    "LOCAL_CONTENT": {
        "penalty_invalid": 0,  # penalty for unmasked invalid MOVE actions
        "masked_node_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "imbalance_pairs": [(70, 69, [1, 3]),
                            (70, 71, [2, 4]),
                            (69, 68, [2, 3]),
                            (73, 72, [2, 3]),
                            (74, 73, [2, 3]),
                            (75, 74, [2, 3])],
        "num_sub_node": 3,  # num of sub waypoints
    },

}
