import gym
import park
import os 
from park.park_to_gym_wrapper import ParkWrapper
from stable_baselines3 import A2C, PPO
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='load_balance')
parser.add_argument('--algo', type=str, default='ppo')
parser.add_argument('--job_distribution', type=str, default='Pareto', choices=["Pareto", "Saw", "Uniform", "CyclicPos", "CyclicNeg", "DriftPos", "DriftNeg", "Constant"])
args = parser.parse_args()

# create gym env from park env
env = park.make(f'{args.env_name}-{args.job_distribution}')
env = ParkWrapper(env, env_name=args.env_name)

# create folders
save_path = f'../../data/park_{args.algo}/{args.env_name}'
os.makedirs(f'../../data/park_{args.algo}', exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/replay_buffers', exist_ok=True)

# rl and save replay buffers
if args.algo == 'ppo':
    algo_class = PPO 
elif args.algo == 'a2c':
    algo_class = A2C
model = algo_class("MlpPolicy", env, verbose=1, save_path=save_path)
model.learn(total_timesteps=10_000)

# test
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

