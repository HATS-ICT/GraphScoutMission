import gym
import numpy as np
from random import randrange, uniform

from graph_scout.envs.utils.agent.agent_cooperative import AgentCoop
from graph_scout.envs.utils.agent.agent_heuristic import AgentHeur


class ScoutMissionStd(gym.Env):
    def __init__(self, **kwargs):
        # 0 setup general elements and containers
        self.configs = {}
        self.agents = None
        self.states = None
        self.map = None

        self.step_counter = 0
        self.done_counter = 0

        # 1 init environment configuration arguments
        # 1.1 init all local/default configs and parse additional arguments
        self._init_env_config(**kwargs)
        # 1.2 load terrain related info (load parsed connectivity & visibility graphs)
        self._load_map_graph()
        # 1.3 generate agent & state instances
        self._init_agent_state()

        # 2 init Multi-branched action gym space & flattened observation gym space
        from gym import spaces
        self.action_space = [[spaces.MultiDiscrete(self.acts.shape())]] * self.states.num
        self.observation_space = [[spaces.Box(low=0.0, high=1.0, shape=(self.states.shape,),
                                              dtype=float)]] * self.states.num

    def reset(self, force=False):
        self.step_counter = 0
        if force:
            self.done_counter = 0
        self.agents.reset()
        self.states.reset()
        self.update()
        self.update_observation()
        return np.array(self.states.obs_list())

    def reset_step_count(self):
        self.step_counter = 0

    def step(self, n_actions):
        self.step_acts = n_actions
        self.step_counter += 1

        # mini step interactions
        _invalid = self.validate_actions()
        self.prep_all_agent_state()
        self.agent_interactions()

        # state update
        self.update()
        self.update_observation()

        # rewards calculation
        # step rewards: stored @self.states.rewards[:, self.step_counter]
        self.get_step_rewards(_invalid)

        # True: if an agent loses all health points (or @max_step)
        n_done = self.get_step_done()
        self.states.done_list = n_done
        if all(done is True for done in n_done):
            # add episodic rewards; stored @self.states.rewards[:, 0]
            self.get_episode_rewards()
            n_rewards = self.states.reward_list(self.step_counter)
            for _a in range(self.states.num):
                n_rewards[_a] += self.states.rewards[_a, 0]
            # update global done counts 
            self.done_counter += 1
        return np.array(self.states.obs_list()), n_rewards, n_done, {}

    def validate_actions(self):
        action_penalty = [0] * self.states.num
        assert self.agents.n_obs == len(self.step_acts), f"[GSMEnv][Step] Unexpected action shape: {self.step_acts}"

        for _index in range(self.agents.n_obs):
            actions = self.step_acts[_index]
            # Check input action if in range of the desired discrete action space
            assert self.action_space[_index].contains(actions), f"[GSMEnv][Step] Action out of range: {actions}"
            _id = self.agents.ids_ob[_index]
            _agent = self.agents.gid[_id]

            # Skip actions for death agents
            if _agent.death:
                continue

            action_move, action_look, action_body = actions
            # Find all first order neighbors of the current node
            _node = self.agents.gid[_id].at_node
            dict_move_node = self.map.get_Gmove_neighbor_dict(_node)
            # Validate actions with masks
            if action_move and (action_move not in dict_move_node):
                # If the action_mask turns on in learning, invalid actions should not appear.
                if self.invalid_masked:
                    assert f"[GSMEnv][Step] action:{action_move} node:{_node} masking:{self.action_mask[_index]}"
                # If the learning process doesn't apply action masking, then the invalid Move should be replaced by 0:"NOOP".
                else:
                    action_move = 0
                    action_penalty[_index] = self.configs["penalty_invalid"]

            # Update state tuple(loc, dir, pos) after executing actions
            if action_move in dict_move_node:
                _node = dict_move_node[action_move]
            dir_anew = self.acts.look[action_look]
            pos_anew = action_body
            # Fill up engage matrix for later use.
            self.engage_mat[_id, -3:] = [dir_anew, pos_anew, _node]
            # Agent 'dir' & 'pos' are updated immediately; Do NOT update 'at_node' at this stage.
            _agent.set_acts(action_move, dir_anew, pos_anew)
            _agent.step_reset()
        return action_penalty

    def prep_all_agent_state(self):
        # [TODO][!!] only support heuristic_blue + lr_red
        # get all other agents' next states before executing mini-steps
        # decision trees for DT agents
        # behavior branches: {FORWARD, ASSIST, RETREAT}

        # DIR POS target selection
        for a_src in self.agents.ids_dt:
            # Skip death agent
            if self.agents.gid[a_src].death:
                continue

            # If current target_agent is None, set new target if possible
            node_src = self.agents.gid[a_src].at_node
            agent_tar = self.agents.gid[a_src].target_agent
            if agent_tar < 0:
                # {target_agent_id: distance}
                list_dist = {}
                n_src = self.engage_mat[a_src, a_src]
                # store all possible {target_aid: dist}
                for a_tar in self.agents.ids_R:
                    n_tar = self.engage_mat[a_tar, a_tar]
                    if self.map.g_view.has_edge(n_src, n_tar):
                        list_dist[a_tar] = self.map.get_Gview_edge_attr_dist(n_src, n_tar)
                # have opposite agents in sight, select the nearest target
                _dir, _pos = [0, 0]
                if any(list_dist):
                    _ids = list(list_dist.keys())
                    _dist = list(list_dist.values())
                    target_dist = min(_dist)
                    target_id = _ids[_dist.index(target_dist)]
                    self.agents.gid[a_src].target_agent = target_id
                    _dir = self.map.get_Gview_edge_attr_dir(n_src, self.engage_mat[target_id, target_id])
                    zone_id = self._get_zone_by_dist(target_dist)
                    if zone_id > 1:
                        _pos = 1
                        self.agents.gid[a_src].change_speed_slow()
                    self.engage_mat[a_src, target_id] = zone_id
                    # [TBD] update buffer
                else:
                    # random acts if no target in range
                    _dir = randrange(self.acts.n_look)
                    _pos = 0
                self.engage_mat[a_src, -3:-1] = [_dir, _pos]
                self.agents.gid[a_src].set_acts(0, _dir, _pos)
            # keep and update current target
            else:
                if self.map.g_view.has_edge(node_src, self.agents.gid[agent_tar].at_node):
                    # maintain current target if in sight
                    continue
                else:
                    # [TBD] buffer
                    self.agents.gid[a_src].target_agent = -1


        # update matrix
        self._update_engage_matrix(team_dt=False)

        # MOVE
        if self.step_counter < self.configs["num_hibernate"]:
            return True
        for a_src in self.agents.ids_dt:
            _agent = self.agents.gid[a_src]
            # Skip death agent
            if _agent.death:
                continue
            if _agent.slow_mode:
                self.engage_mat[a_src, -1] = _agent.move_slow_mode_prep()
            else:
                self.engage_mat[a_src, -1] = _agent.move_en_route_prep()
        return False

    # update health points for all agents
    def agent_interactions(self):
        # update health and damage points for all agents
        _step_damage = self.configs["damage_single"]
        n_mini_step = self.configs["num_sub_step"]
        mid_step = int(n_mini_step / 2)

        for sub_step in range(n_mini_step):
            # change anchor node from move_src to move_tar
            if sub_step == mid_step:
                # make the real MOVE action here after a certain mini-step
                self._update_agents_locations()
            # check engage matrix to get the engagement for each sub-step
            for _r in self.agents.ids_R:
                for _b in self.agents.ids_B:
                    # check reed_blue engage flags
                    token_r_b = self.engage_mat[_r, _b]
                    if token_r_b:
                        _damage = self._get_prob_by_src_tar(_r, _b, token_r_b)
                        if _damage:
                            # shooting successful
                            self.agents.gid[_r].damage_given(_step_damage)
                            self.agents.gid[_b].damage_taken(_step_damage)
                        else:
                            # even though no shooting, marks disturbing
                            self.agents.gid[_r].disturbing()
                    # check blue_red engage flags
                    token_b_r = self.engage_mat[_b, _r]
                    if token_b_r and self._get_prob_by_src_tar(_b, _r, token_b_r):
                        self.agents.gid[_b].damage_given(_step_damage)
                        self.agents.gid[_r].damage_taken(_step_damage)

        return False

    def _get_prob_by_src_tar(self, u_id, v_id, zone_id) -> bool:
        # input int zone_id > 0
        if zone_id == 4:
            # overlapping agents are guaranteed to engage (any dir & pos)
            return True
        u_node = self.engage_mat[u_id, u_id]
        v_node = self.engage_mat[v_id, v_id]
        u_dir = self.engage_mat[u_id, -3]
        u_pos = self.engage_mat[u_id, -2]
        v_pos = self.engage_mat[v_id, -2]
        pos_u_v = self._get_pos_u_v(u_pos, v_pos)
        # get the engagement probability
        prob_raw = self.map.get_Gview_prob_by_dir_pos(u_node, v_node, u_dir, pos_u_v)
        prob_add = prob_raw + self.zones[zone_id]["prob_add"]
        # prob_fin = prob_add * self.zones[zone_id]["prob_mul"]
        # generate a random number in the range of [0., 1.]
        flag_rand = uniform(0., 1.)
        # determine if cause damage during this interaction
        return flag_rand < prob_add

    def _update_agents_locations(self):
        for _id in range(self.agents.n_all):
            node_anew = self.engage_mat[_id, -1]
            if self.engage_mat[_id, _id] != node_anew:
                self.engage_mat[_id, _id] = node_anew
                self.agents.gid[_id].at_node = node_anew
        self._reset_engage_matrix()
        self._update_engage_matrix(team_ob=True, team_dt=True)

    def _update_engage_matrix(self, team_ob=True, team_dt=True):
        # [TODO][!!] only support heuristic_blue + lr_red
        # each agent only have (at most) ONE active target agent
        # ===> one zone token per row
        if team_ob:
            for _r in self.agents.ids_R:
                _u = self.engage_mat[_r, _r]
                max_zone, max_id = [0, 0]
                for _b in self.agents.ids_B:
                    _v = self.engage_mat[_b, _b]
                    _zone = self._get_zone_token(_u, _v)
                    if _zone and _zone > max_zone:
                        max_zone = _zone
                        max_id = _b
                if max_zone:
                    self.engage_mat[_r, max_id] = max_zone

        if team_dt:
            for _b in self.agents.ids_B:
                _u = self.engage_mat[_b, _b]
                max_zone, max_id = [0, 0]
                for _r in self.agents.ids_R:
                    _v = self.engage_mat[_r, _r]
                    _zone = self._get_zone_token(_u, _v)
                    if _zone and _zone > max_zone:
                        max_zone = _zone
                        max_id = _b
                if max_zone:
                    self.engage_mat[_b, max_id] = max_zone

        return False

    def _reset_engage_matrix(self):
        # engage_mat token clean up -> reset to 0 for the next step
        for _r in self.agents.ids_R:
            for _b in self.agents.ids_B:
                self.engage_mat[_r, _b] = 0
                self.engage_mat[_b, _r] = 0
        return False

    # update local states after all agents finished actions
    def update(self):
        self._reset_engage_matrix()
        # [TODO] new action masking
        # generate binary matrices for pair-wised inSight and inRange indicators
        if self.invalid_masked:
            for _a in range(self.agents.n_obs):
                # update action masking
                # self.action_mask[a_id] = np.zeros(sum(tuple(self.action_space[_r].nvec)), dtype=bool)
                a_id = self.agents.ids_ob[_a]
                # masking invalid movements on the given node
                acts = set(self.acts.move.keys())
                valid = set(self.map.get_all_action_Gmove(self.agents.gid[a_id]) + [0])
                invalid = [_ for _ in acts if _ not in valid]
                for masking in invalid:
                    self.action_mask[_a][masking] = True
        return False

    def update_observation(self):
        """ ===> setup custom observation shape & value
        # set values in the teammate and opposite observation slots
        # 1.0 at node
        # 0.5 inside dangerous zone
        # 0.3 inside cautious zone
        # 0.2 in sight
        # 0.1 within machine gun range (team blue only)
        """
        obs_value = [0.1, 0.2, 0.3, 0.5, 1.0]
        death_decay = 0.7
        self.states.reset_obs_per_step()

        # update elements in team red slot
        for r_id in self.agents.ids_R:
            if self.agents.gid[r_id].death:
                continue
            _dir, _pos, _src = self.engage_mat[r_id, -3:]
            # only check target Standing pos
            _pos_edge = self._get_pos_u_v(_pos, 0)
            neighbors = self.map.get_Gview_neighbor_by_dir_pos(_src, _dir, _pos_edge)
            for _tar in neighbors:
                value = obs_value[self._get_zone_obs(_src, _tar)]
                # set argmax on each node
                if value > self.states.obs_R[_tar - 1]:
                    self.states.obs_R[_tar - 1] = value
            self.states.obs_R[_src - 1] = obs_value[-1]

        # update elements in team blue slot
        node_end = self.configs["field_boundary_node"]  # machine gun coverage area [0 to node_end]
        # add min_val for nodes covered by blue's machine gun squad 
        self.states.obs_B[0:node_end] = obs_value[0]
        for b_id in self.agents.ids_B:
            _dir, _pos, _src = self.engage_mat[b_id, -3:]
            if self.agents.gid[b_id].death:
                self.states.obs_B[_src - 1] = obs_value[-1] * death_decay
                continue
            _pos_edge = self._get_pos_u_v(_pos, 0)
            neighbors = self.map.get_Gview_neighbor_by_dir_pos(_src, _dir, _pos_edge)
            for _tar in neighbors:
                value = obs_value[self._get_zone_obs(_src, _tar)]
                if value > self.states.obs_B[_tar - 1]:
                    self.states.obs_B[_tar - 1] = value
            self.states.obs_B[_src - 1] = obs_value[-1]

        return False

    def _get_zone_token(self, node_src, node_tar) -> int:
        """ zone token lookup for engage matrix updates
        # check self.configs["engage_token"] for more details
        # if there is an edge in the visibility FOV graph;
        # [!] no self-loop in the visibility graph, check if two agents are on the same node first
        """
        obs_range = [self.zones[3]['dist'], self.zones[2]["dist"]]
        if node_src == node_tar:
            return 4  # overlap
        if self.map.g_view.has_edge(node_src, node_tar):
            dist = self.map.get_Gview_edge_attr_dist(node_src, node_tar)
            # -1 indicates there is no visibility range limitation
            max_range = self.configs["sight_range"]  # or self.zones[1]['dist']
            if max_range and dist > max_range:
                return 0
            return self._get_zone_by_dist(dist)
        return 0

    def _get_zone_obs(self, node_src, node_tar) -> int:
        # simple zone token retrieval without checking corner cases
        dist = self.map.get_Gview_edge_attr_dist(node_src, node_tar)
        return self._get_zone_by_dist(dist)

    def _get_zone_by_dist(self, dist) -> int:
        # ranges = [dangerous_zone_boundary=50, cautious_zone_boundary=150]
        obs_range = [self.zones[3]['dist'], self.zones[2]["dist"]]
        return 1 if dist > obs_range[1] else (2 if dist > obs_range[0] else 3)

    def _get_pos_u_v(self, pos_u, pos_v):
        # source/target in range {0:"Stand", 1:"Prone"} -> [0,1,2,3]
        return pos_u + pos_u + pos_v

    def get_step_rewards(self, prev_rewards):
        rewards = prev_rewards
        rew_cfg = self.rew_cfg["step"]
        if rew_cfg["rew_step_pass"]:
            return rewards
        for _index in range(self.agents.n_ob):
            _agent = self.agents.gid[self.agents.ids_ob[_index]]
            if _agent.engaged_step:
                rewards[_index] += rew_cfg["rew_step_slow"]
                _dmg_taken = _agent.dmg_step_taken
                _dmg_given = _agent.dmg_step_given
                if _dmg_given:
                    if _dmg_taken < _dmg_given:
                        rewards[_index] += rew_cfg["rew_step_adv"]
                    else:
                        rewards[_index] += rew_cfg["rew_step_dis"]
        return rewards

    def get_episode_rewards(self):
        # gather final states
        rewards = [0] * self.states.num
        rew_cfg = self.rew_cfg["episodic"]
        if rew_cfg["rew_ep_pass"]:
            return rewards
        # [TODO]
        return rewards

    def get_step_done(self):
        # reach to max_step
        if self.step_counter >= self.max_step:
            return [True] * self.agents.n_obs
        # all team_Blue or team_Red agents got terminated
        if all([self.agents.gid[_id].death for _id in self.agents.ids_B]) or all(
                [self.agents.gid[_id].death for _id in self.agents.ids_R]):
            return [True] * self.agents.n_obs
        # death => done for each observing agent (early termination)
        return [self.agents.gid[_id].death for _id in self.agents.ids_ob]

    # load configs and update local defaults
    def _init_env_config(self, **kwargs):
        from graph_scout.envs.utils.config.default_configs import init_setup as env_cfg
        """ set default env config values if not specified in outer configs """
        from copy import deepcopy
        self.configs = deepcopy(env_cfg["INIT_ENV"])
        self.rew_cfg = deepcopy(env_cfg["INIT_REWARD"])
        self.log_cfg = {}

        _config_local = env_cfg["INIT_LOCAL"]
        _config_all = list(_config_local.keys()) + list(self.configs.keys())
        _reward_step = list(self.rew_cfg["step"].keys())
        _reward_epic = list(self.rew_cfg["episode"].keys())
        _log = list(env_cfg["INIT_LOG"].keys())

        # loading outer args and overwrite local default configs
        for key, value in kwargs.items():
            if key in _config_all:
                self.configs[key] = value
            elif key in _reward_step:
                self.rew_cfg["step"][key] = value
            elif key in _reward_epic:
                self.rew_cfg["episode"][key] = value
            elif key in _log:
                self.log_cfg[key] = value
            else:
                raise ValueError(f"[GSMEnv][Init] Invalid config: \'{key}:{value}\'")

        # eazy access vars -> most frequently visited args
        self.n_red = self.configs["num_red"]
        self.n_blue = self.configs["num_blue"]
        self.max_step = self.configs["max_step"]

        # set defaults bool vars if not specified
        for key in _config_local:
            if key in self.configs:
                continue
            else:
                self.configs[key] = _config_local[key]
            # load content if True
            if self.configs[key]:
                _new_key = env_cfg["LOCAL_TRANS"][key]
                self.configs[_new_key] = env_cfg["LOCAL_CONTENT"][key]
        self.invalid_masked = self.configs["masked_act"]
        if self.invalid_masked:
            self.action_mask = [[np.zeros(sum(tuple(self.action_space[0].nvec)), dtype=bool)]] * self.agents.n_obs

        # check agent configs: must have attributes to init & reset
        # set -> self.configs["agents_init"]

        # setup init logs if not provided
        self.logger = self.configs["log_on"] if "log_on" in self.configs else False
        if self.logger:
            self.log_cfg["root_path"] = self.configs["env_path"]
            for key in _log:
                if key not in self.log_cfg:
                    self.log_cfg[key] = env_cfg["INIT_LOG"][key]
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
        self.agents = AgentManager(self.n_red, self.n_blue, self.configs["health_red"], self.configs["health_blue"],
                                   **self.configs["agents_init"])
        _id, _name, _team = self.agents.get_observing_agent_info()
        # <default obs> all agents have identical observation shape: (teammate slot + opposite slot)
        _obs_shape = self.map.get_graph_size()

        self.states = StateManager(self.agents.n_obs, _obs_shape, self.max_step, _id, _name, _team)

        # lookup dicts for all action branches
        from action_lookup import ActionBranched as actsEval
        self.acts = actsEval()

        # engagement matrix for all agent-pairs + [dir, pos, target_node_after_move] per agent
        self.engage_mat = np.zeros((self.agents.n_all, self.agents.n_all + 3), dtyp=int)
        for _a in range(self.agents.n_all):
            _node, _dir, _pos = self.agents.gid[_a].get_geo_tuple()
            self.engage_mat[_a, _a] = _node
            self.engage_mat[_a, -3:] = [_dir, _pos, _node]
        self.zones = self.configs["engage_range"]

        # memory of heuristic agents [branch, signal_e, signal_h, target_agent, graph_dist, target_node]
        _dts = len(self.agents.ids_dt)
        self.buffer_mat = np.zeros((self.configs["buffer_size"], _dts, 5), dtyp=int)
        self.buffer_ptr = [-1] * _dts

    def _engage_step_reset(self):
        self.engage_mat[:, :-3] = 0
        for _a in range(self.agents.n_all):
            self.engage_mat[_a, _a] = self.engage_mat[_a, -1]
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


class AgentManager:
    def __init__(self, n_red=0, n_blue=0, HP_r=100, HP_b=100, **agent_config):
        # list of all agent object instances (sorted by global_id)
        self.gid = list()

        # numbers of agents
        self.n_all = n_red + n_blue
        self.n_obs = 0

        # args for resetting agents:
        self.list_init = list()  # [node, 0(motion), dir, posture, health]
        self.dict_path = dict()  # designated paths

        # gid lookup lists
        self.ids_ob = list()  # index of learning agents
        self.ids_dt = list()  # index of heuristic agents
        self.ids_R = list()  # team_id == 0 (red)
        self.ids_B = list()  # team_id == 1 (blue)

        self._load_init_configs(n_red, n_blue, HP_r, HP_b, **agent_config)

    def _load_init_configs(self, n_red, n_blue, HP_red, HP_blue, **agent_config):
        # agent global_id is indexing from 0
        g_id = 0
        default_HP = [HP_red, HP_blue]
        for _d in agent_config:
            # dict key: agent name (i.e. "red_0", "blue_1")
            _name = _d

            # dict values for each agent: team, type, node/path, acts, HP and tokens
            _type = agent_config[_d]["type"]
            _team = agent_config[_d]["team_id"]
            _dir = agent_config[_d]["direction"]
            _pos = agent_config[_d]["posture"]
            _HP = default_HP[_team]

            # learning agents
            if _type == "RL":
                _node = agent_config[_d]["node"]
                self.gid.append(AgentCoop(global_id=g_id, name=_name, team_id=_team, health=_HP,
                                          node=_node, direction=_dir, posture=_pos,
                                          _learning=agent_config[_d]["is_lr"],
                                          _observing=agent_config[_d]["is_ob"]))
                self.ids_ob.append(g_id)
                self.list_init.append([_node, 0, _dir, _pos, _HP])
                self.n_obs += 1

            # pre-determined agents
            elif _type == "DT":
                _path = agent_config[_d]["path"]
                self.gid.append(AgentHeur(global_id=g_id, name=_name, team_id=_team, health=_HP,
                                          path=_path, direction=_dir, posture=_pos))
                self.ids_dt.append(g_id)
                self.list_init.append([0, 0, _dir, _pos, _HP])
                self.dict_path[g_id] = _path

            # [TBD] default agent & behavior
            else:
                raise ValueError(f"[GSMEnv][Agent] Unexpected agent type: {_type}")

            # update team lookup lists
            if _team:
                self.ids_B.append(g_id)
            else:
                self.ids_R.append(g_id)

            g_id += 1

        # [TBD] add default RL agents if not enough agent_init_configs were provided
        # or just raise error
        if len(self.ids_R) != n_red or len(self.ids_B) != n_blue:
            raise ValueError("[GSMEnv][Agent] Not enough init configs are provided.")

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


class StateManager:
    def __init__(self, num=0, shape=0, max_step=0, ids=None, names=None, teams=None):
        self.num = num
        self.shape = shape
        # agent lookup
        self.a_id = [0] if ids is None else ids
        # dict keys
        self.name_list = [0] if names is None else names
        self.team_list = [0] if teams is None else teams  # {0: "red", 1: "blue"}
        # dict values
        # observation slots for teammate and opposite [single copy]
        self.obs_R = np.zeros(shape)
        self.obs_B = np.zeros(shape)
        self.rewards = np.zeros((num, max_step + 1))
        self.done_list = np.zeros(num, dtype=bool)

    def reset(self):
        self.reset_step()
        self.rewards[:] = 0.
        self.done_list[:] = False

    def reset_step(self):
        self.obs_R[:] = 0.
        self.obs_B[:] = 0.

    def obs_list(self):
        list_obs = []
        for index in range(self.num):
            if self.team_list[index]:
                # from team blue's perspective
                list_obs.append(self.obs_B.tolist() + self.obs_R.tolist())
            else:
                # from team red's perspective
                list_obs.append(self.obs_R.tolist() + self.obs_B.tolist())
        return list_obs

    def reward_list(self, step):
        return self.rewards[:, step].tolist()

    def dump_dict(self, step=0):
        _dict_obs = {}
        _dict_rew = {}
        _dict_done = {}
        for _id in range(self.num):
            _key = self.name_list[_id]
            _dict_obs[_key] = np.concatenate(self.obs_B, self.obs_R) if self.team_list[_id] else np.concatenate(
                self.obs_R, self.obs_B)
            _dict_rew[_key] = self.rewards[_id, step]
            _dict_done[_key] = self.done_list[_id]
        return _dict_obs, _dict_rew, _dict_done
