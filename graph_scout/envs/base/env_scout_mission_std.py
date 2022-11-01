import gym
import numpy as np
from random import randrange, uniform


class ScoutMissionStd(gym.Env):
    def __init__(self, **kwargs):
        #0 setup general elements and containers
        self.configs = {}
        self.agents = None
        self.states = None
        self.map = None

        self.step_counter = 0
        self.done_counter = 0

        #1 init environment config arguments
        #1.1 init all local/default configs and parse additional arguments
        self._init_env_config(**kwargs)
        #1.2 load env terrain geo-info (loading connectivity & visibility graphs)
        self._load_map_graph()
        #1.3 get agent & state instances
        self._init_agent_state()
        
        #2 init Multi-branched action gym space & flattened observation gym space
        from gym import spaces
        self.action_space = [[spaces.MultiDiscrete(self.acts.shape())]] * self.states.num
        self.observation_space = [[spaces.Box(low=0.0, high=1.0, shape=(self.states.shape,), dtype=np.float64)]] * self.states.num

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
        self.step_acts = n_actions
        assert len(self.step_acts) == self.states.num, f"[GSMEnv][Step] Invalid action shape {n_actions}"
        self.step_counter += 1

        # mini step interactions
        _mini_engage = self._engage_step_reset()
        for _step in range(self.mini_steps):
            for _id in self.agents.ids_ob:
                _mini_engage = self.view_source_target(_id, _step, _mini_engage)
        _engage = self.agent_state_anew()

        # state update
        _states_flags = self.update()
        self.update_observation()
        
        # rewards calculation
        self._step_rewards(_engage, _mini_engage)
        # set to True if a agent loses its all health points (or @max_step)
        n_done = self._get_step_done()
        self.states.done_list = n_done
        if all(done is True for done in n_done):
            # add episodic rewards
            self._episode_rewards() # store in self.states.rewards[_agents, 0]
            n_rewards = self.states.reward_list(self.step_counter)
            for _a in range(self.states.num):
                n_rewards[_a] += self.states.rewards[_a, 0]
            # update global done counts 
            self.done_counter += 1
        return np.array(self.states.obs_list()), n_rewards, n_done, {}

    def update_observation(self):
        # set values in the teammate and opposite observation slots
        # 1.0 at node
        # 0.5 in dangerious zone
        # 0.25 in cautious zone
        # 0.1~0.2 in machine gun range
        for _id in self.agents.ids_R:
            _loc, _dir, _pos = self.agents.gid[_id].get_geo_tuple()


        for _id in self.agents.ids_B:
            _loc, _dir, _pos = self.agents.gid[_id].get_geo_tuple()


    def agent_state_anew(self):
        action_penalty = [0] * self.states.num
        _action_stay_penalty = self.configs["penalty_stay"] if "penalty_stay" in self.configs else 0

        for _index, actions in enumerate(self.step_acts):
            # check input action if in range of the desired discrete action space
            assert self.action_space[_index].contains(actions), f"{actions}: action out of range"
            _id = self.agents.ids_ob[_index]
            if self.agents.gid[_id].death:
                continue
            action_move, action_look, action_body = actions
            # find all 1st ordered neighbors of the current node
            _node = self.agents.gid[_id].at_node
            prev_node, list_neighbors, list_acts = self.map.get_all_states_by_node(_node)
            # validate actions with masks
            if action_move != 0 and action_move not in list_acts:
                # if the action_mask turns on in the learning, invalid actions should not appear.
                if self.invalid_masked:
                    assert f"[ActError] action{action_move} node{prev_node} masking{self.action_mask[_index]}"
                # if the learning process doesn't have action masking, then invalid Move should be replaced by NOOP.
                else:
                    action_move = 0
                    action_penalty[_index] = self.configs["penalty_invalid"]

            # make branched actions
            if action_move == 0:
                if _action_stay_penalty:
                    action_penalty[_index] += _action_stay_penalty
            elif action_move in list_acts:
                _node = list_neighbors[list_acts.index(action_move)]
            self.agents.gid[_id].set_states([_node, action_move, self.acts.look[action_look], action_body])
        return action_penalty

    # update local states after all agents finished actions
    def update(self):
        # generate binary matrices for pair-wised inSight and inRange indicators
        R_see_B = np.zeros((self.n_red, self.n_blue), dtype=np.bool_)
        R_engage_B = np.zeros((self.n_red, self.n_blue), dtype=np.bool_)
        B_see_R = np.zeros((self.n_blue, self.n_red), dtype=np.bool_)
        B_engage_R = np.zeros((self.n_blue, self.n_red), dtype=np.bool_)
        R_nodes = [0] * self.n_red
        R_overlay = [False] * self.n_red
        if self.invalid_masked:
            self.action_mask = [np.zeros(sum(tuple(self.action_space[_].nvec)), dtype=np.bool_)
                                for _ in range(len(self.obs_agents))]
        for _r in range(self.n_red):
            node_r, dir_r = self.team_red[_r].get_pos_dir()
            R_nodes[_r] = node_r
            for _b in range(self.n_blue):
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
        for _s in range(self.n_red - 1):
            for _t in range(_s + 1, self.n_red):
                if R_nodes[_s] == R_nodes[_t]:
                    R_overlay[_s] = True
                    R_overlay[_t] = True

        # update states for all agents in team red
        look_dir_shape = len(self.configs["ACT_LOOK_DIR"])
        _obs_self_dir = self.obs_token["obs_dir"]
        _state_R_dir = []
        # get looking direction encodings for all red agents
        if _obs_self_dir:
            _state_R_dir = np.zeros((self.n_red, look_dir_shape))
            for _r in range(self.n_red):
                _, _dir = self.team_red[_r].get_pos_dir()
                _state_R_dir[_r, (_dir - 1)] = 1

        # get next_move_dir encodings for all blue agents
        _state_B_next = np.zeros((self.n_blue, look_dir_shape))
        for _b in range(self.n_blue):
            _route = self.team_blue[_b].get_route()
            _index = self.team_blue[_b].get_index()
            _dir = self.routes[_route].list_next[_index]
            _state_B_next[_b, (_dir - 1)] = 1

        ''' update state for each agent '''
        # condition: multi-hot pos encodings for self, 'team_blue' and 'team_red' in the shape of len(G.all_nodes())
        pos_obs_size = self.map.get_graph_size()
        # concatenate state_self + state_blues (+ state_reds)
        for _r in range(self.n_red):
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
            for _b in range(self.n_blue):
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
                for _agent in range(self.n_red):
                    if _agent != _r:
                        _state_R[R_nodes[_agent] - 1] = 1
                _state += _state_R

            # update the local state attribute
            self.states[_r] = _state
        
        return R_see_B, R_engage_B, B_see_R, B_engage_R, R_overlay

    # update health points for all agents
    def agent_interaction(self):
        # update health and damage points for all agents
        _step_damage = self.configs["engage_behavior"]["damage"]
        for _b in self.agents.ids_B:
            for _r in self.agents.ids_R:
                if self.mat_engage[_r, _b]:
                    self.agents.gid[_r].cause_damage(_step_damage)
                    self.agents.gid[_b].take_damage(_step_damage)
                if self.mat_engage[_b, _r]:
                    self.agents.gid[_r].take_damage(_step_damage)
            # update end time for blue agents
            if self.agents.gid[_b].get_end_step() > 0:
                continue
            _damage_taken_blue = self.configs["init_health_blue"] - self.agents.gid[_b].health
            if _damage_taken_blue >= self.configs["threshold_damage_2_blue"]:
                self.agents.gid[_b].set_end_step(self.step_counter)

    def _step_rewards(self, rews, mini_rews):
        rewards = rews
        if self.rewards["step"]["reward_step_on"] is False:
            return rewards
        for _a in self.agents.ids_ob:
            rewards[_a] += get_step_overlay(mini_rews[_a], **self.rewards["step"])
            for agent_b in range(self.n_blue):
                rewards[agent_r] += get_step_engage(r_engages_b=self.mat_engage[agent_r, agent_b],
                                                    b_engages_r=self.mat_engage[agent_b, agent_r],
                                                    team_switch=False, **self.rewards["step"])
        return rewards

    def _episode_rewards(self):
        # gather final states
        _HP_full_r = self.configs["init_health_red"]
        _HP_full_b = self.configs["init_health_blue"]
        _threshold_r = self.configs["threshold_damage_2_red"]
        _threshold_b = self.configs["threshold_damage_2_blue"]

        _health_lost_r = [_HP_full_r - self.team_red[_r].get_health() for _r in range(self.n_red)]
        _damage_cost_r = [self.team_red[_r].damage_total() for _r in range(self.n_red)]
        _health_lost_b = [_HP_full_b - self.team_blue[_b].get_health() for _b in range(self.n_blue)]
        _end_step_b = [self.team_blue[_b].get_end_step() for _b in range(self.n_blue)]

        rewards = [0] * self.n_red
        if self.rewards["episode"]["reward_episode_on"] is False:
            return rewards
        rewards = get_episode_reward_team(_health_lost_r, _health_lost_b, _threshold_r, _threshold_b,
                                          _damage_cost_r, _end_step_b, **self.rewards["episode"])

        # #-- If any Red agent got terminated, the whole team would not receive the episode rewards
        # if any([_health_lost_r[_r] > _threshold_r for _r in range(self.n_red)]):
        #     return rewards
        # for agent_r in range(self.n_red):
        #     for agent_b in range(self.n_blue):
        #         rewards[agent_r] += get_episode_reward_agent(_health_lost_r[agent_r], _health_lost_b[agent_b],
        #                                                      _threshold_r, _threshold_b, _damage_cost_r[agent_r],
        #                                                      _end_step_b[agent_b], **self.rewards["episode"])
        return rewards

    def _get_step_done(self):
        # reach to max_step
        if self.step_counter >= self.max_step:
            return [True] * self.agents.n_obs
        # all team_Blue or team_Red agents got terminated
        if all([self.agents.gid[_id].death for _id in self.agents.ids_B]) or 
            all([self.agents.gid[_id].death for _id in self.agents.ids_R]):
            return [True] * self.agents.n_obs
        # death == done for each observing agent (early termination)
        return [self.agents.gid[_id].death for _id in self.agents.ids_ob]

    def _is_in_sight(self, source_node, target_node, source_dir):
        """ field of view check
            if there is an edge in the visibility FOV graph;
                if so, check if it is inside the sight range
            <!> no self-loop in the visibility graph for now, check if two agents are on the same node first
        """
        if source_node == target_node:
            return True
        if self.map.g_view.has_edge(source_node, target_node):
            _distance = self.map.get_Gview_edge_attr_dist(source_node, target_node, source_dir)
            # -1 indicates there is no visibility edge in the 's_dir' direction
            _range = self.configs["sight_range"]
            if _distance == -1:
                return False
            if _range < 0 or _distance < _range:
                return True
        return False

    # load configs and update local defaults
    def _init_env_config(self, **kwargs):
        from graph_scout.envs.utils.config import default_configs as env_cfg
        """ set default env config values if not specified in outer configs """
        from copy import deepcopy
        self.configs = deepcopy(env_cfg.INIT_CONFIGS)
        self.rewards = deepcopy(env_cfg.INIT_REWARDS)
        # fast access for frequently visited args
        self.n_red = self.configs["num_red"]
        self.n_blue = self.configs["num_blue"]
        self.max_step = self.configs["max_step"]
        self.mini_steps = self.configs["mini_n_step"]

        _config_local_args = env_cfg.INIT_CONFIGS_LOCAL
        _config_args = _config_local_args + list(self.configs.keys())
        _reward_step_args = list(self.rewards["step"].keys())
        _reward_done_args = list(self.rewards["episode"].keys())
        _log_args = list(env_cfg.INIT_LOGS.keys())

        # loading outer args and overwrite env configs
        for key, value in kwargs.items():
            # assert env_cfg.check_args_value(key, value)
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
                self.configs[key] = self.configs["damage_maximum"] * self.n_red
            elif key == "act_masked":
                self.configs[key] = env_cfg.ACT_MASKED["mask_on"]
        # setup penalty for invalid action in unmasked conditions
        self.invalid_masked = self.configs["act_masked"]
        if self.invalid_masked is False:
            self.configs["penalty_invalid"] = env_cfg.ACT_MASKED["unmasked_invalid_action_penalty"]

        # check init_red init configs, must have attribute "pos":[str/tuple] for resetting agents
        self.configs["init_red"] = env_cfg.check_agent_init("red", self.n_red, self.configs["init_red"])
        # check init_blue, must have attribute "route":[str] used in loading graph files. default: '0'
        self.configs["init_blue"] = env_cfg.check_agent_init("blue", self.n_blue, self.configs["init_blue"])
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
                        self.logs[item] = env_cfg.INIT_LOGS[item]
        return True

    # load terrain graphs. [*]executable after calling self._init_env_config()
    def _load_map_graph(self):
        from graph_scout.envs.data.file_manager import load_graph_files
        from graph_scout.envs.data.terrain_graph import MapInfo
        # load graphs
        self.map = MapInfo()
        self.map = load_graph_files(env_path=self.configs["env_path"], map_lookup=self.configs["map_id"])
        # [TODO] call generate_files if parsed data not exist

    # initialize all agents. [*] executable after calling self._init_env_config()
    def _init_agent_state(self):
        self.agents = AgentManager(self.configs["n_red"], self.configs["n_blue"], self.configs["agent_init"])
        _id, _name, _team = self.agents.get_observing_agent_info()
        
        # <default obs> all agents have identical observation shape: (teammate slot + opposite slot)
        _obs_shape = self.map.get_graph_size()
        # _obs_shape = self.map.get_graph_size() - len(self.configs["nodes_masked"])
        self.states = StateManager(self.agents.n_obs, _obs_shape, self.max_step, _id, _name, _team)
        # self.states.dump_dict()
        from action_lookup import ActionBranched as actsEval
        self.acts = actsEval()
        # engagement matrix for all agent pairs
        self._engage_step_reset()

    def _engage_step_reset(self):
        self.mat_engage = np.zeros((self.agents.n_all, self.agents.n_all), dtyp=bool)
        return [0] * self.states.num

    def _log_step_states(self):
        return self.states.dump_dict() if self.logger else []

    def render(self, mode='human'):
        pass

    # delete local instances
    def close(self):
        if self.agents is not None:
            del self.agents
        if self.states is not None:
            del self.states
        if self.map is not None:
            del self.map

from graph_scout.envs.utils.agent.agent_cooperative import AgentCoop as agentRL
from graph_scout.envs.utils.agent.agent_heuristic import AgentHeur as agentDT


class AgentManager():
    def __init__(self, n_red=0, n_blue=0, **agent_config):
        self.n_all = n_red + n_blue
        self.n_obs = 0
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
            # dict key: agent name (i.e. "red_0", "blue_1")
            _name = _d
            # dict values for each agent
            _team = agent_config[_d["team_id"]]
            _HP = agent_config[_d["health"]]
            _dir = agent_config[_d["direction"]]
            _pos = agent_config[_d["posture"]]
            _type = agent_config[_d["type"]]
            # learning agents
            if _type == "RL":
                _node = agent_config[_d["node"]]
                self.gid.append(agentRL(global_id=g_id, name=_name, team_id=_team, health=_HP, 
                                            node=_node, direction=_dir, posture=_pos,
                                            _learning=agent_config[_d["learn"]],
                                            _observing=agent_config[_d["sense"]]))
                self.ids_ob.append(g_id)
                self.list_init.append([_node, 0, _dir, _pos, _HP])
                self.n_obs += 1
            # pre-determined agents
            elif _type == "DT":
                _path = agent_config[_d["path"]]
                self.gid.append(agentDT(global_id=g_id, name=_name, team_id=_team, health=_HP, 
                                            path=_path, direction=_dir, posture=_pos))
                self.ids_dt.append(g_id)
                self.list_init.append([0, 0, _dir, _pos, _HP])
                self.dict_path[g_id] = _path
            # [TBD] default agent & hebavior
            else:
                raise ValueError(f"[GSMEnv][Agent] Unexpected agent type: {_typr}")
            # update team lookup lists [only works for two team scenarios 0 or 1]
            if _team:
                self.ids_B.append(g_id)
            else:
                self.ids_R.append(g_id)
            g_id += 1
        # [TBD] add default RL agents if not enough agent_init_configs were provided
        # raise error atm
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
    def __init__(self, num=1, shape=0, max_step=1, ids=[], names=[], teams=[]):
        self.num = num
        self.shape = shape
        # agent lookup
        self.a_id = ids
        # dict keys
        self.name_list = names
        self.team_list = teams # team_id {"red": 0, "blue", 1}
        # dict values
        # observation slots for teammate and opposite [single copy]
        self.obs_R = np.zeros(shape)
        self.obs_B = np.zeros(shape)
        self.rewards = np.zeros((num, max_step))
        self.done_list = [False] * num
        
    def reset(self, num=1, max_step=1):
        self.obs_R = np.zeros(shape)
        self.obs_B = np.zeros(shape)
        self.rewards = np.zeros((num, max_step))
        self.done_list = [False] * num

    def obs_list(self):
        list_obs = []
        for index in range(self.num):
            if self.team_list[index]:
                # team red's perspective
                list_obs.append(obs_B.tolist() + obs_R.tolist())
            else:
                # team blue's perspective
                list_obs.append(obs_R.tolist() +_obs_B.tolist())
        return list_obs

    def reward_list(self, step):
        return self.rewards[:, step].tolist()

    def.dump_dict(self, step=0):
        _dict_obs = {}
        _dict_rew = {}
        _dict_done = {}
        for _id in range(self.num):
            _dict_obs[self.name_list[_id]] = np.concatenate(self.obs_B, self.obs_R) if self.team_list[_id] else np.concatenate(self.obs_R, self.obs_B)
            _dict_rew[self.name_list[_id]] = self.rewards[_id, step]
            _dict_done[self.name_list[_id]] = self.done_list[_id]
        return _dict_obs, _dict_rew, _dict_done

