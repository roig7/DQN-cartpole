import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """Neural Net for Deep-Q learning Model."""
        model = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_dim=self.state_size),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        next_states = []
        targets_f = []

        for state, action, reward, next_state, done in minibatch:
            states.append(np.reshape(state, (self.state_size,)))  # Append reshaped state to list
            if not done:
                next_states.append(np.reshape(next_state, (self.state_size,)))  # Append reshaped next_state to list

        # Convert lists to NumPy arrays for prediction
        states_np = np.array(states)
        next_states_np = np.array(next_states)

        # Predict all next Q-values in one network pass if there are next states to predict
        if len(next_states) > 0:
            q_next = self.model.predict(next_states_np)
            q_next_max = np.amax(q_next, axis=1)

        # Calculate targets for each experience in the minibatch
        index = 0
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                target = reward + self.gamma * q_next_max[index]
                index += 1
            # Update the target for the action taken
            target_f = self.model.predict(np.array([np.reshape(state, (self.state_size,))]))
            target_f[0][action] = target
            targets_f.append(target_f[0])

        # Perform a single batch update to the model
        self.model.train_on_batch(states_np, np.array(targets_f))

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Create the env and init the agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size=state_size, action_size=action_size)

episodes = 1000  # Define the number of episodes for training
for e in range(episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, info, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Store transition in replay buffer
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {e + 1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

        # Train the agent with the experience of the episode
        agent.replay()

    # Optionally save the model
#      if (e + 1) % 50 == 0:
#        agent.save(f'cartpole_model_{e+1}.h5')
