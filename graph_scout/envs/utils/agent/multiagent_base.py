class GSMAgent:
    def __init__(self, global_id=0, name="A0", team_id=0, node=0,
                motion=0, direction=0, posture=0, health=100, _death=False):
        # basic info
        self.id = global_id
        self.name = name
        self.team = team_id
        # map info
        self.at_node = node
        # action states
        self.motion = motion
        self.direction = direction
        self.posture = posture
        # interactive args
        self.health = health
        self.death = _death

    @property
    def death(self):
        return self._death

    @death.setter
    def death(self, value):
        if value != bool(value):
            raise TypeError("[GSMEnv][Agent] value must be a bool")
        self._death = value
    
    # fast update without verifying values
    def set_states(self, num_list):
        self.at_node = num_list[0]
        self.motion = num_list[1]
        self.direction = num_list[2]
        self.posture = num_list[3]

    def get_acts(self):
        return [self.motion, self.direction, self.posture]

    def get_geo_tuple(self):
        return [self.at_node, self.direction, self.posture]

    # health value is greater or equal to 0
    def take_damage(self, num_deduction):
        if num_deduction < self.health:
            self.health -= num_deduction
        else:
            self.health = 0
            self.death = True

    def reset(self, list_states, health=100, _death=False):
        self.set_states(list_states)
        self.health = health
        self.death = _death
