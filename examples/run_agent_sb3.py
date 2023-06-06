import gym
import park
import os 
from park.park_to_gym_wrapper import ParkWrapper
from stable_baselines3 import A2C, PPO
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='abr_sim')
parser.add_argument('--algo', type=str, default='a2c')
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--distribution', type=str, default='Pareto', choices=["default", "Pareto", "Saw", "Uniform", "CyclicPos", "CyclicNeg", "DriftPos", "DriftNeg", "Constant"])
args = parser.parse_args()

# create gym env from park env
if 'load_balance' in args.env_name:
    env = park.make(f'{args.env_name}-{args.distribution}')
else:
    env = park.make(f'{args.env_name}')
env = ParkWrapper(env, env_name=args.env_name)

# create folders
save_path = f'../../data/park_{args.algo}/{args.env_name}_{args.distribution}_{args.n_steps}'
os.makedirs(f'../../data/park_{args.algo}', exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/replay_buffers', exist_ok=True)

# rl and save replay buffers
if args.algo == 'ppo':
    algo_class = PPO 
elif args.algo == 'a2c':
    algo_class = A2C
model = algo_class("MlpPolicy", env, verbose=1, ent_coef=1.0, ent_decay_num_steps = 10_000, n_steps=args.n_steps, max_grad_norm=10.0, save_path=save_path)
model.learn(total_timesteps=100_000_000)

# test
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

