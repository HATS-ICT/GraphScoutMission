from multiagent_base import GSMAgent


class AgentHeur(GSMAgent):
    def __init__(self, global_id=1, name="B0", team_id=1, node=0,
                motion=-1, direction=0, posture=0, health=100, _death=False,
                path=None, index=0, mini_steps=4, wait_steps=1):
        super().__init__(global_id, name, team_id, node,
                        motion, direction, posture, health, _death)
        # pre-designed route and the pointer for current location
        self.index = index
        self.route = path
        self.target_node = 0
        self.init_source_target()
        # interactive args
        self.target_agent = 0
        self.slow_mode = False
        # {step_num} >= 1: agent moves on an edge at a speed slower than normal.
        # ==> {N} mini-steps/segments (or {N-1} sub-waypoints)
        self.slow_step = int(mini_steps) if mini_steps > 1 else 1
        self._slow_count = self.slow_step
        # {step_num} >= 1: agent visits the target for {N} steps to claim a success.
        self.stay_step = int(wait_steps) if wait_steps > 1 else 1
        self._stay_count = self.stay_step

    @property
    def route(self):
        return self._route

    @route.setter
    def route(self, node_list):
        if node_list is None:
            node_list = [super().at_node]
        self._route = node_list
        self._route_max = len(node_list) - 1
        if self.index > self._route_max:
            self.index = self._route_max

    def change_route(self, node_list, index=0):
        self.index = index
        self.route = node_list
        if self.at_node != self.route[0]:
            raise ValueError("[GSMEnv][Agent] route source node does not match.")

    def reset(self, list_states, path, index=0, health=100, _death=False):
        super().reset(list_states, health, _death)
        self.index = index
        self.route = path
        self.init_source_target()
        self.target_agent = 0
        self.slow_mode = False
        self._slow_count = self.slow_step
        self._stay_count = self.stay_step

    # set moving speed alongwith the posture lookup token
    def change_speed_slow(self, posture=1):
        self.slow_mode = True
        self.posture = posture

    def change_speed_fast(self, posture=0):
        self.slow_mode = False
        self._slow_count = self.slow_step
        self.posture = posture

    # move to the next node in the designated path
    def move_en_route(self) -> bool:
        if self.index < self._route_max:
            self.index += 1
            self.at_node = self._route[self.index]
            return True
        return False

    # move for one time step; finish in one or multiple steps
    def move_sub_nodes(self) -> bool:
        if self._slow_count:
            self._slow_count -= 1
            return True
        else:
            self.move_en_route()
            self._slow_count = self.slow_step
            return False

    def if_at_main_nodes(self) -> bool:
        return self._slow_count == self.slow_step

    # set final target location
    def init_source_target(self):
        if self._route_max:
            self.at_node = self._route[self.index]
            self.target_node = self._route[-1]

    # fast accessing current & next nodes with no boundary checks: self.index < max
    def get_source_target(self):
        return self.at_node, self._route[self.index + 1]

    def if_path_end(self):
        return self.index == self._route_max

    # check if the agent has staying at the target node for enough steps
    def done_scout(self):
        if self._stay_count:
            if self.target_node == self.at_node:
                self._stay_count -= 1
            return False
        return True
