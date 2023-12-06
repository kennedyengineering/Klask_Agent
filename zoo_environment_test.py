from zoo_environment import CustomEnvironment
from pettingzoo.test import parallel_api_test
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1000000)
    check_env(env)