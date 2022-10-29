from multiagent_base import GSMAgent


class AgentCoop(GSMAgent):
    def __init__(self, global_id=0, name="R0", team_id=0, node=0,
                motion=0, direction=0, posture=0, health=100, _death=False,
                _learning=True, _observing=True):
        super().__init__(global_id, name, team_id, node,
                        motion, direction, posture, health, _death)
        # ineractive args
        self.damage_total = 0
        self.disturb_total = 0
        # RL control args
        self.is_learning = _learning
        self.is_observing = _observing

    # A binary token for active learning or forzen status
    @property
    def is_learning(self):
        return self._learning

    @is_learning.setter
    def is_learning(self, value):
        if value != bool(value):
            raise TypeError("[GSMEnv][Agent] value must be a bool")
        self._learning = value

    # A binary token for hidden/blind agents
    @property
    def is_observing(self):
        return self._observing

    @is_observing.setter
    def is_observing(self, value):
        if value != bool(value):
            raise TypeError("[GSMEnv][Agent] value must be a bool")
        self._observing = value

    def reset(self, list_states, health=100, _death=False, _learning=True, _observing=True):
        super().reset(list_states, health, _death)
        self.damage_total = 0
        self.disturb_total = 0
        self.is_learning = _learning
        self.is_observing = _observing

    # major engagements only
    def cause_damage(self, num_point):
        self.damage_total += num_point

    # + minor interactions
    def disturbing(self, num_point=1):
        self.disturb_total += num_point
