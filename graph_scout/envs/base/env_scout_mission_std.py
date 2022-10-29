import numpy as np
import gym

from graph_scout.envs.data.file_manager import load_graph_files
from graph_scout.envs.data.terrain_graph import MapInfo

from action_lookup import ActionBranched as actsEval
# from graph_scout.envs.utils.config.default_setups import init_setups as env_setup


class ScoutMissionStd(gym.Env):
    def __init__(self, max_step=40, n_red=2, n_blue=2, **kwargs):
        #0 setup general elements and containers
        self.configs = {}
        self.agents = AgentManager()
        self.states = StateManager() #dict of act, obs, rew, done
        self.map = MapInfo()

        #1 init environment config arguments
        #1.1 init all local/default configs and parse additional arguments
        self._init_env_config(n_red, n_blue, **kwargs)
        #1.2 load env terrain geo-info (loading connectivity & visibility graphs)
        self._load_map_graph()
        #1.3 get agent & state instances
        self._init_agent_state()
        # <default obs> all agents have identical observation shape: (teammate slot + opposite slot)
        self.state_shape = env_setup.get_state_shape(self.map.get_graph_size(), self.mask_region)

        #2 init gym env spaces
        from gym import spaces
        self.action_space = [spaces.MultiDiscrete(actsEval.shape) for _ in range(self.agents.num)]
        self.observation_space = [spaces.Box(low=0, high=1, shape=(self.state_shape,), dtype=np.int8) for _ in range(self.agents.num)]

        self.max_step = max_step
        self.step_counter = 0
        self.done_counter = 0

    def reset(self, force=False):
        self.step_counter = 0
        if force:
            self.done_counter = 0
        self.agents.reset()
        self.states.reset()
        self.update()
        return np.array(self.states.obs_list())

    def reset_step_count(self):
        self.step_counter = 0

    def step(self, n_actions):
        assert len(n_actions) == self.agents.num, f"[GSMEnv][Step] Invalid action shape {n_actions}"
        # store previous state for logging if logger is 'on'
        # prev_obs = self._log_step_prev()
        self.step_counter += 1

        # mini step interactions
        for _step in range(self.mini_steps):
            for _agent in self.agent_gid:
            _step_engage = self.view_source_target(_step)
            _engage = self.agent_interaction(n_actions, _step_engage)

        # state update
        anew_states_flags = self.update()
        
        # rewards update
        self._step_rewards()
        # set Done=True if agents lost all health points or reach max step
        n_done = self._get_step_done()
        # update done counts and add episodic rewards
        if all(done is True for done in n_done):
            self._episode_rewards() # store in self.states.rewards[_agents, 0]
            n_rewards = self.states.reward_list(self.step_counter)
            for _a in range(self.agents.num):
                n_rewards[_a] += self.states.rewards[_a, 0]
            self.done_counter += 1
        return np.array(self.states.obs_list()), n_rewards, n_done, {}

    def _take_action_agents(self, n_actions):
        action_penalty = [0] * self.num_red
        _action_stay_penalty = self.configs["penalty_stay"] if "penalty_stay" in self.configs else 0

        for agent_i, actions in enumerate(n_actions):
            # check input action if in range of the desired discrete action space
            assert self.action_space[agent_i].contains(actions), f"{actions}: action out of range"
            if self.team_red[agent_i].death:
                continue
            action_move, action_turn = actions
            # find all 1st ordered neighbors of the current node
            agent_encoding = self.team_red[agent_i].agent_code
            prev_node, list_neighbors, list_acts = self.map.get_all_states_by_node(agent_encoding)
            # validate actions with masks
            if action_move != 0 and action_move not in list_acts:
                # if the action_mask turns on in the learning, invalid actions should not appear.
                if self.invalid_masked:
                    assert f"[ActError] action{action_move} node{prev_node} masking{self.action_mask[agent_i]}"
                # if the learning process doesn't have action masking, then invalid Move should be replaced by NOOP.
                else:
                    action_move = 0
                    action_penalty[agent_i] = self.configs["penalty_invalid"]

            # only make 'Turn' when the move action is NOOP
            if action_move == 0:
                if _action_stay_penalty:
                    action_penalty[agent_i] += _action_stay_penalty
                agent_dir = self.team_red[agent_i].agent_dir
                if action_turn == 1:
                    agent_dir = env_setup.act.TURN_L[agent_dir]
                elif action_turn == 2:
                    agent_dir = env_setup.act.TURN_R[agent_dir]
                self.team_red[agent_i].agent_dir = agent_dir
            # make 'Move' and then 'Turn' actions
            elif action_move in list_acts:
                _node = list_neighbors[list_acts.index(action_move)]
                _code = self.map.get_name_by_index(_node)
                _dir = action_move
                # 'Turn' condition for turning left or right
                if action_turn:
                    _dir = env_setup.act.TURN_L[action_move] if action_turn == 1 else env_setup.act.TURN_R[action_move]
                self.team_red[agent_i].set_location(_node, _code, _dir)
        return action_penalty

    def _take_action_blue(self, n_actions=None):
        for agent_i in range(self.num_blue):
            if self.team_blue[agent_i].is_frozen():
                continue
            _route = self.team_blue[agent_i].get_route()
            _idx = self.step_counter % self.routes[_route].get_route_length()
            _node, _code, _dir = self.routes[_route].get_location_by_index(_idx)
            self.team_blue[agent_i].update_index(_idx, _node, _code, _dir)
        # return [0] * self.num_blue

    # update local states after all agents finished actions
    def update(self):
        # generate binary matrices for pair-wised inSight and inRange indicators
        R_see_B = np.zeros((self.num_red, self.num_blue), dtype=np.bool_)
        R_engage_B = np.zeros((self.num_red, self.num_blue), dtype=np.bool_)
        B_see_R = np.zeros((self.num_blue, self.num_red), dtype=np.bool_)
        B_engage_R = np.zeros((self.num_blue, self.num_red), dtype=np.bool_)
        R_nodes = [0] * self.num_red
        R_overlay = [False] * self.num_red
        if self.invalid_masked:
            self.action_mask = [np.zeros(sum(tuple(self.action_space[_].nvec)), dtype=np.bool_)
                                for _ in range(len(self.obs_agents))]
        for _r in range(self.num_red):
            node_r, dir_r = self.team_red[_r].get_pos_dir()
            R_nodes[_r] = node_r
            for _b in range(self.num_blue):
                node_b, dir_b = self.team_blue[_b].get_pos_dir()
                R_see_B[_r, _b] = self.is_in_sight(node_r, node_b, dir_r)
                R_engage_B[_r, _b] = self.is_in_range(node_r, node_b, dir_r)
                B_see_R[_b, _r] = self.is_in_sight(node_b, node_r, dir_b)
                B_engage_R[_b, _r] = self.is_in_range(node_b, node_r, dir_b)
            # update action masking
            if self.invalid_masked:
                # self.action_mask[_r] = np.zeros(sum(tuple(self.action_space[_r].nvec)), dtype=np.bool_)
                mask_idx = self.obs_agents.index(self.team_red[_r].get_id())
                # masking invalid movements on the given node
                acts = set(local_action_move.keys())
                valid = set(self.map.get_actions_by_node(self.team_red[_r].get_encoding()) + [0])
                invalid = [_ for _ in acts if _ not in valid]
                for masking in invalid:
                    self.action_mask[mask_idx][masking] = True

        # update overlap list for team red
        for _s in range(self.num_red - 1):
            for _t in range(_s + 1, self.num_red):
                if R_nodes[_s] == R_nodes[_t]:
                    R_overlay[_s] = True
                    R_overlay[_t] = True

        # update states for all agents in team red
        look_dir_shape = len(env_setup.ACT_LOOK_DIR)
        _obs_self_dir = self.obs_token["obs_dir"]
        _state_R_dir = []
        # get looking direction encodings for all red agents
        if _obs_self_dir:
            _state_R_dir = np.zeros((self.num_red, look_dir_shape))
            for _r in range(self.num_red):
                _, _dir = self.team_red[_r].get_pos_dir()
                _state_R_dir[_r, (_dir - 1)] = 1

        # get next_move_dir encodings for all blue agents
        _state_B_next = np.zeros((self.num_blue, look_dir_shape))
        for _b in range(self.num_blue):
            _route = self.team_blue[_b].get_route()
            _index = self.team_blue[_b].get_index()
            _dir = self.routes[_route].list_next[_index]
            _state_B_next[_b, (_dir - 1)] = 1

        ''' update state for each agent '''
        # condition: multi-hot pos encodings for self, 'team_blue' and 'team_red' in the shape of len(G.all_nodes())
            pos_obs_size = self.map.get_graph_size()
            # concatenate state_self + state_blues (+ state_reds)
            for _r in range(self.num_red):
                # add self position one-hot embedding
                _state = [0] * pos_obs_size
                _state[R_nodes[_r] - 1] = 1
                # add self direction if True
                if _obs_self_dir:
                    _state += _state_R_dir[_r, :].tolist()
                # add self interaction indicators
                if self.obs_token["obs_sight"]:
                    _state += R_see_B[_r, :].tolist()
                if self.obs_token["obs_range"]:
                    _state += R_engage_B[_r, :].tolist()

                # add team blue info
                _state_B = [0] * pos_obs_size
                for _b in range(self.num_blue):
                    _node, _ = self.team_blue[_b].get_pos_dir()
                    _state_B[_node - 1] = 1
                    # add next move dir
                    _state_B += _state_B_next[_b, :].tolist()
                if self.obs_token["obs_sight"]:
                    _state_B += B_see_R[:, _r].tolist()
                if self.obs_token["obs_range"]:
                    _state_B += B_engage_R[:, _r].tolist()
                _state += _state_B

                # add teammates pos if True
                if self.obs_token["obs_team"]:
                    _state_R = [0] * pos_obs_size
                    for _agent in range(self.num_red):
                        if _agent != _r:
                            _state_R[R_nodes[_agent] - 1] = 1
                    _state += _state_R

                # update the local state attribute
                self.states[_r] = _state
        
        return R_see_B, R_engage_B, B_see_R, B_engage_R, R_overlay

    # update health points for all agents
    def agent_interaction(self, R_engage_B, B_engage_R):
        # update health and damage points for all agents
        _step_damage = env_setup.INTERACT_LOOKUP["engage_behavior"]["damage"]
        for _b in range(self.num_blue):
            for _r in range(self.num_red):
                if R_engage_B[_r, _b]:
                    self.team_red[_r].damage_add(_step_damage)
                    self.team_blue[_b].take_damage(_step_damage)
                if B_engage_R[_b, _r]:
                    self.team_red[_r].take_damage(_step_damage)
            # update end time for blue agents
            if self.team_blue[_b].get_end_step() > 0:
                continue
            _damage_taken_blue = self.configs["init_health_blue"] - self.team_blue[_b].get_health()
            if _damage_taken_blue >= self.configs["threshold_damage_2_blue"]:
                self.team_blue[_b].set_end_step(self.step_counter)

    def _step_rewards(self, penalties, R_engage_B, B_engage_R, R_overlay):
        rewards = penalties
        if self.rewards["step"]["reward_step_on"] is False:
            return rewards
        for agent_r in range(self.num_red):
            rewards[agent_r] += get_step_overlay(R_overlay[agent_r], **self.rewards["step"])
            for agent_b in range(self.num_blue):
                rewards[agent_r] += get_step_engage(r_engages_b=R_engage_B[agent_r, agent_b],
                                                    b_engages_r=B_engage_R[agent_b, agent_r],
                                                    team_switch=False, **self.rewards["step"])
        return rewards

    def _episode_rewards(self):
        # gather final states
        _HP_full_r = self.configs["init_health_red"]
        _HP_full_b = self.configs["init_health_blue"]
        _threshold_r = self.configs["threshold_damage_2_red"]
        _threshold_b = self.configs["threshold_damage_2_blue"]

        _health_lost_r = [_HP_full_r - self.team_red[_r].get_health() for _r in range(self.num_red)]
        _damage_cost_r = [self.team_red[_r].damage_total() for _r in range(self.num_red)]
        _health_lost_b = [_HP_full_b - self.team_blue[_b].get_health() for _b in range(self.num_blue)]
        _end_step_b = [self.team_blue[_b].get_end_step() for _b in range(self.num_blue)]

        rewards = [0] * self.num_red
        if self.rewards["episode"]["reward_episode_on"] is False:
            return rewards
        rewards = get_episode_reward_team(_health_lost_r, _health_lost_b, _threshold_r, _threshold_b,
                                          _damage_cost_r, _end_step_b, **self.rewards["episode"])

        # #-- If any Red agent got terminated, the whole team would not receive the episode rewards
        # if any([_health_lost_r[_r] > _threshold_r for _r in range(self.num_red)]):
        #     return rewards
        # for agent_r in range(self.num_red):
        #     for agent_b in range(self.num_blue):
        #         rewards[agent_r] += get_episode_reward_agent(_health_lost_r[agent_r], _health_lost_b[agent_b],
        #                                                      _threshold_r, _threshold_b, _damage_cost_r[agent_r],
        #                                                      _end_step_b[agent_b], **self.rewards["episode"])
        return rewards

    def _get_step_done(self):
        # reach to max_step
        if self.step_counter >= self.max_step:
            return [True] * self.num_red
        # all Blue agents got terminated
        if all([self.team_blue[_b].get_health() <= 0 for _b in range(self.num_blue)]):
            return [True] * self.num_red
        # done for each Red agent
        return [self.team_red[_r].get_health() <= 0 for _r in range(self.num_red)]

    def is_in_sight(self, source_node, target_node, source_dir):
        """ field of view check
            if there is an edge in the visibility FOV graph;
                if so, check if it is inside the sight range
            <!> no self-loop in the visibility graph for now, check if two agents are on the same node first
        """
        if source_node == target_node:
            return True
        if self.map.g_view.has_edge(source_node, target_node):
            _distance = self.map.get_edge_attr_vis_fov_by_idx(s_idx, t_idx, s_dir)
            # -1 indicates there is no visibility edge in the 's_dir' direction
            _range = env_setup.INTERACT_LOOKUP["sight_range"]
            if _distance == -1:
                return False
            if _range < 0 or _distance < _range:
                return True
        return False

    def is_in_range(self, s_idx, t_idx, s_dir):
        """ engage behavior indicator check
            if there is an edge in visibility graph;
                if so, check if the distance is below the engaging range limit
        """
        if s_idx == t_idx:
            return True
        if self.map.g_vis.has_edge(s_idx, t_idx):
            _distance = self.map.get_edge_attr_vis_fov_by_idx(s_idx, t_idx, s_dir)
            if _distance == -1:
                return False
            if _distance < env_setup.INTERACT_LOOKUP["engage_range"]:
                return True
        return False

    # load configs and update local defaults [!!] need to know num_red and num_blue
    def _init_env_config(self, **kwargs):
        """ set default env config values if not specified in outer configs """
        from copy import deepcopy
        self.configs = deepcopy(env_setup.INIT_CONFIGS)
        self.obs_token = deepcopy(env_setup.OBS_TOKEN)
        self.rewards = deepcopy(env_setup.INIT_REWARDS)

        _config_local_args = env_setup.INIT_CONFIGS_LOCAL
        _config_args = _config_local_args + list(self.configs.keys())
        _obs_shape_args = list(self.obs_token.keys())
        _reward_step_args = list(self.rewards["step"].keys())
        _reward_done_args = list(self.rewards["episode"].keys())
        _log_args = list(env_setup.INIT_LOGS.keys())

        # loading outer args and overwrite env configs
        for key, value in kwargs.items():
            # assert env_setup.check_args_value(key, value)
            if key in _config_args:
                self.configs[key] = value
            elif key in _obs_shape_args:
                self.obs_token[key] = value
            elif key in _reward_step_args:
                self.rewards["step"][key] = value
            elif key in _reward_done_args:
                self.rewards["episode"][key] = value
            elif key in _log_args:
                self.logs[key] = value
            else:
                print(f"Invalid config argument \'{key}:{value}\'")

        # set local defaults if not predefined or loaded
        for key in _config_local_args:
            if key in self.configs:
                continue
            if key == "threshold_damage_2_red":
                self.configs[key] = self.configs["damage_maximum"]
            elif key == "threshold_damage_2_blue":
                # grant blue agents a higher damage threshold when more reds on the map
                self.configs[key] = self.configs["damage_maximum"] * self.num_red
            elif key == "act_masked":
                self.configs[key] = env_setup.ACT_MASKED["mask_on"]
        # setup penalty for invalid action in unmasked conditions
        self.invalid_masked = self.configs["act_masked"]
        if self.invalid_masked is False:
            self.configs["penalty_invalid"] = env_setup.ACT_MASKED["unmasked_invalid_action_penalty"]

        # check init_red init configs, must have attribute "pos":[str/tuple] for resetting agents
        self.configs["init_red"] = env_setup.check_agent_init("red", self.num_red, self.configs["init_red"])
        # check init_blue, must have attribute "route":[str] used in loading graph files. default: '0'
        self.configs["init_blue"] = env_setup.check_agent_init("blue", self.num_blue, self.configs["init_blue"])
        # get all unique routes. (blue agents might share patrol route)
        self.configs["route_lookup"] = list(set(_blue["route"] for _blue in self.configs["init_blue"]))

        # setup log inits if not provided
        if "log_on" in self.logs:
            self.logger = self.logs["log_on"]
            # turn on logger if True
            if self.logger is True:
                self.logs["root_path"] = self.configs["env_path"]
                for item in _log_args[1:]:
                    if item not in self.logs:
                        self.logs[item] = env_setup.INIT_LOGS[item]
        return True

    # load terrain graphs. [*]executable after _init_env_config
    def _load_map_graph(self):
        # load graphs
        self.map = load_graph_files(env_path=self.configs["env_path"], map_lookup=self.configs["map_id"])
        # [TODO] call generate_files if non-exist

    # initialize all agents. [*] executable after _init_env_config
    def _init_agent_state(self):
        for idx, init_red in enumerate(self.configs["init_red"]):
            r_uid = idx
            learn = init_red["learn"] if "learn" in init_red else True
            if learn is True:
                self.obs_agents.append(r_uid)
            self.team_red.append(AgentRed(_uid=r_uid, _learn=learn))

        for idx, init_blue in enumerate(self.configs["init_blue"]):
            b_uid = (idx + self.num_red)
            learn = init_blue["learn"] if "learn" in init_blue else False
            if learn is True:
                self.obs_agents.append(b_uid)
            b_route = self.configs["route_lookup"].index(init_blue["route"])  # int: index in the route lookup list
            self.team_blue.append(AgentBlue(_uid=b_uid, _learn=learn, _route=b_route))

    # reset agents to init status for each new episode
    def _reset_agents(self):
        HP_red = self.configs["init_health_red"]
        for idx, init_red in enumerate(self.configs["init_red"]):
            r_code = env_setup.get_default_red_encoding(idx, init_red["pos"])
            r_node = self.map.get_index_by_name(r_code)
            r_dir = env_setup.get_default_dir(init_red["dir"])
            self.team_red[idx].reset(_node=r_node, _code=r_code, _dir=r_dir, _health=HP_red)

        HP_blue = self.configs["init_health_blue"]
        for idx, init_blue in enumerate(self.configs["init_blue"]):
            b_route = self.team_blue[idx].get_route()
            b_index = init_blue["idx"]  # int: index of the position on the given route
            b_node, b_code, b_dir = self.routes[b_route].get_location_by_index(b_index)
            self.team_blue[idx].reset(_node=b_node, _code=b_code, _dir=b_dir, _health=HP_blue, _index=b_index, _end=-1)

    def _log_step_states(self):
        return self.states.dump() if self.logger else []

    def render(self, mode='human'):
        pass

    # delete local instances
    def close(self):
        del self.agents
        del self.states
        del self.map

from graph_scout.envs.utils.agent.agent_cooperative import AgentCoop as agentRL
from graph_scout.envs.utils.agent.agent_heuristic import AgentHeur as agentDT


class AgentManager():
    def __init__(self, n_red=1, n_blue=1, **agent_config):
        self.num = n_red + n_blue
        self.gid = list() # list of all agent object instances
        self.list_init = list() # saved init args: [node, 0(motion), dir, posture, health]
        self.dict_path = {}
        self.ids_ob = list() # index of RL agents
        self.ids_dt = list() # index of decision tree agents
        self.ids_R = list() # team_id == 0
        self.ids_B = list() # team_id == 1
        self._load_init_configs(n_red, n_blue, configs)

    def _load_init_configs(self, n_red, n_blue, **agent_config):
        g_id = 0
        for _d in agent_config:
            _name = _d["name"]
            _team = _d["team_id"]
            _HP = _d["health"]
            _dir = _d["direction"]
            _pos = _d["posture"]
            _type = _d["type"]
            if _type == "RL":
                _node = _d["node"]
                self.gid.append(agentRL(global_id=g_id, name=_name, team_id=_team, health=_HP, 
                                            node=_node, direction=_dir, posture=_pos,
                                            _learning=_d["learn"], _observing=_d["sense"]))
                self.ids_ob.append(g_id)
                self.list_init.append([_node, 0, _dir, _pos, _HP])
            elif _type == "DT":
                _path = _d["path"]
                self.gid.append(agentDT(global_id=g_id, name=_name, team_id=_team, health=_HP, 
                                            path=_path, direction=_dir, posture=_pos))
                self.ids_dt.append(g_id)
                self.list_init.append([0, 0, _dir, _pos, _HP])
                self.dict_path[g_id] = _path
            else:
                # [TBD] default agent & hebavior
                raise ValueError(f"[GSMEnv][Agent] Unexpected agent type: {_typr}")
            # update team lookups [only works for two team scenarios 0 or 1]
            if _t:
                self.ids_B.append(g_id)
            else:
                self.ids_R.append(g_id)
            g_id += 1
        # add default RL agents if n_red or n_blue is greater than the num_configs
        # [TBD] 
        if len(self.ids_R) != n_red or len(self.ids_B) != n_blue:
            raise ValueError("[GSMEnv][Agent] Not enough agent init configs are provided.")

    def reset(self):
        for _id in self.ids_ob:
            self.gid[_id].reset(list_states=self.list_init[_id][0:-1],
                                health=self.list_init[_id][-1])
        for _id in self.ids_dt:
            self.gid[_id].reset(list_states=self.list_init[_id][0:-1],
                                health=self.list_init[_id][-1],
                                path=self.dict_path[_id])

    def get_observing_agent_info(self):
        list_name = []
        list_team = []
        for _id in self.ids_ob:
            list_name.append(self.gid[_id].name)
            list_team.append(self.gid[_id].team_id)
        return self.ids_ob, list_name, list_team

    def close(self):
        del self.gid


class StateManager():
    def __init__(self, num=1, max_step=1):
        self.num = num
        # dict keys
        self.name_list = []

        # dict values
        self.team_list = [] # team_id {"red": 0, "blue", 1}
        self.obs_R = []
        self.obs_B = []
        self.rewards = None
        self.done_list = []
        # observation slots for teammate and opposite [single copy]
        
    def reset(self, num=1, max_step=1):
        self.rewards = np.zeros((num, max_step))
        self.done_list = [[False]] * num
        self.obs_R = []
        self.obs_B = []

    def obs_list(self):
        list_obs = []
        for index in range(self.num):
            if team_list[index]:
                list_obs.append(obs_B + obs_R)
            else:
                list_obs.append(obs_R +_obs_B)
        return list_obs

    def reward_list(self, step):
        return self.rewards[:, step].tolist()

    def to_dict(self):
        dict_act = {}
        dict_obs = {}
        dict_rew = {}
        dict_done = {}
        for _index, name in enumerate(self.name_list):
            dict_act[name] = self.acts_list[_index]
            dict_obs

