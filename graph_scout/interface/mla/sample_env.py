import gym
import graph_scout
from random import randrange
import argparse


def environment_sample_run(configs):
    env_name = configs.env_name
    max_ep = configs.max_episode
    local_configs = {}
    if hasattr(configs, "env_path"):
        local_configs["env_path"] = configs.env_path
    if hasattr(configs, "max_step"):
        local_configs["max_step"] = configs.max_step

    # => Step 1. make env
    env = gym.make(env_name, **local_configs)
    # print inits
    print(f"Env created with configs: {env.configs}\n")

    # episode loop
    for ep in range(max_ep):
        print(f"\n#####===> Episode: {ep + 1} of {max_ep}\n")

        # ==> Step 2. initial 'reset' before running 'step' functions
        obs = env.reset()
        branch_act_range = env.acts.shape()
        _id, _name, _team = env.agents.get_observing_agent_info()
        print(f"\n Agent names (dictionary keys): {_name}")
        print(f"Init observations: {env.states.obs_full}")

        # issue with the current terrain map setup
        # print("[!!] Undesirable moving directions 86-69:")
        # print(f"Valid actions from node: {86} {env.map.get_Gmove_action_node_dict(86)}")
        # print(f"Valid actions from node: {69} {env.map.get_Gmove_action_node_dict(69)}")

        # step main loop
        for i in range(env.max_step):
            # random actions
            step_acts = []
            for _id in range(len(env.agents.ids_ob)):
                act_per_agent = []
                for _act in branch_act_range:
                    act_per_agent.append(randrange(_act))
                step_acts.append(act_per_agent)

            # ===> Step 3. execute 'env.step(actions)' for [env.max_step] times in an episode then 'env.reset()'
            list_obs, list_reward, list_done, _ = env.step(step_acts)

            # [!] get the dictionary's for RLlib training
            dict_obs, dict_rew, dict_done = env.states.dump_dict()
            print(f"###==> Step:{env.step_counter} of {env.max_step} | act:{step_acts} rew:{dict_rew}")
            # print(f"Obs: {dict_obs}")

            # early stop
            if all(list_done):
                for _a in env.agents.gid:
                    print(f"Agent_{_a.id}: {_a.name} Node:{_a.get_geo_tuple()} HP:{_a.health}/{_a.health_max}")
                print(f"Rewards for all steps: {env.states.rewards}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic configs
    parser.add_argument('--env_name', type=str, default='graphScoutMission-v0', help='gym env entry point')
    parser.add_argument('--env_path', type=str, default='./', help='path to the env root')
    parser.add_argument('--max_episode', type=int, default=1, help='number of episodes')
    parser.add_argument('--max_step', type=int, default=20, help='number of steps per episode')

    config = parser.parse_args()
    environment_sample_run(config)
