import gym
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# CONSTANTS
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 5_000

# Exploration settings
epsilon = 1  # not a constant, will decay
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


ep_rewards = [-200]


class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = tf.keras.Sequential([
            Dense(24, input_shape=(6,), activation='relu'),
            Dense(24, activation='relu'),
            Dense(3, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    def update_replay_memory(self, transition):
        #print(f"Adding to replay memory: {transition}")
        self.replay_memory.append(transition)

    # Trains main network every step during episodes
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        future_qs_list = self.target_model.predict(new_current_states)

        #print(f"Current states: {current_states.shape}, New states: {new_current_states.shape}")

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        #print(f"Training on X: {np.array(X).shape}, y: {np.array(y).shape}")
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        reshaped_state = np.array(state).reshape(-1, *state.shape)
        #print(f"Getting Qs for state: {reshaped_state.shape}")
        return self.model.predict(reshaped_state)[0]


class Env:
    RETURN_IMAGES = True
    OBSERVATION_SPACE_VALUES = (6,)
    ACTION_SPACE_SIZE = 3

    def __init__(self):
        self.env = gym.make('Acrobot-v1')

    def reset(self):
        self.episode_step = 0
        observation, _ = self.env.reset()  
        #print(f"Raw reset observation: {observation}")  # Debugging statement
        observation = np.array(observation, dtype=np.float32)
        #print(f"Formatted reset observation: {observation.shape}")
        if self.RETURN_IMAGES:
            self.env.render()
        return observation

    def step(self, action):
        self.episode_step += 1
        new_observation, reward, done, truncated, _ = self.env.step(action)
        #print(f"Raw step observation: {new_observation}")  # Debugging statement
        new_observation = np.array(new_observation, dtype=np.float32)
        if self.RETURN_IMAGES:
            self.env.render()
        done = done or truncated
        return new_observation, reward, done


agent = DQNAgent()
env = Env()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, 3)

        new_state, reward, done = env.step(action)
        episode_reward += reward
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

agent.model.save('main_model.h5')
agent.target_model.save('target_model.h5')

# Plotting the results
plt.plot(ep_rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards per Episode')
plt.show()
