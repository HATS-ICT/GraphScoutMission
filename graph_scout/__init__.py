from gym.envs.registration import register

register(
    id='graphScoutMission-v0',
    entry_point='graph_scout.envs.base:ScoutMission',
    max_episode_steps=100,
)

