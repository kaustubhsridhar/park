import park

env = park.make('load_balance')

obs = env.reset()
done = False

while not done:
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
    print(reward)
    exit()