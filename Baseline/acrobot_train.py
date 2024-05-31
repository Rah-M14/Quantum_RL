import gym
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
model=load_model('main_model.h5')

env=gym.make('Acrobot-v1')

episode_rewards=[]

for episode in range(100):
    obs=env.reset()
    done=False
    ep_reward=0
    while not done:
        obs=np.array(obs).reshape(-1, obs.shape[0])
        action=np.argmax(model.predict(obs)[0])
        obs,reward,done,_=env.step(action)
        env.render()
        ep_reward+=reward
    print(f"reward at episode: {episode} is {ep_reward}")
    episode_rewards.append(ep_reward)
env.close()

print(f"average reward over {episode+1} episodes is {sum(episode_rewards)/len(episode_rewards)}")