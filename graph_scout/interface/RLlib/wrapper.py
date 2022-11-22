import os
import ray
import gym
from graph_scout.envs.base.env_scout_mission_std import ScoutMissionStd
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune.registry import register_env


class ScoutMissionWrapper(ScoutMissionStd, MultiAgentEnv):
    def __init__(self, **env_configs):
        super().__init__(**env_configs)
        self.agent_gid, self.agent_names, _ = super().agents.get_observing_agent_info()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        dict_obs = {}
        for index, a_id in enumerate(self.agent_names):
            dict_obs[a_id] = obs[index]
        return dict_obs

    def step(self, actions_dict):
        action_list = list(actions_dict.values())
        _, _, list_done = super.step(action_list)
        dict_obs, dict_rew, dict_done = super().states.dump_dict()
        dict_done['__all__'] = all(list_done)
        return dict_obs, dict_rew, dict_done, {}


def env_creator(env_config):
    return ScoutMissionWrapper(**env_config)  # return an env instance


if __name__ == "__main__":
    env_name = "GraphScoutMission"
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    register_env(env_name, lambda config: env_creator(config))
    config = (
        PPOConfig()
        .rollouts(num_rollout_workers=4, rollout_fragment_length=512)
        .training(
            train_batch_size=512,
            lr=3e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .environment(env=env_name, clip_actions=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"time_steps_total": 100},
        checkpoint_freq=10,
        local_dir="~/Workspace/local_results/logs/" + env_name,
        config=config.to_dict(),
    )
