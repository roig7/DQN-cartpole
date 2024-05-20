import pandas as pd
# from finrl.config_tickers import DOW_30_TICKER
# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# from finrl.meta.preprocessor.preprocessors import FeatureEngineer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from collections import deque
import random
import gym
from gym import spaces
import os
import pickle
from tensorflow.python.keras.losses import Huber
from tensorflow.keras.layers import Dropout, ReLU, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K, backend
from tensorflow.keras.callbacks import Callback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Suppress most TensorFlow  logs (0 = all logs, 3 = FATAL only)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress deprecated function warnings
import logging
import time

from tensorflow.summary import create_file_writer

# Create a log directory with a timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/{}'.format(timestamp)
writer = create_file_writer(log_dir)



data = pd.read_csv('preprocessData.csv')
print(data.head())
# %%
print(data.shape)
# %% md
# 2. Process the data
# %%
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 10000
# WINDOW_SIZE = 10
TRANSACTION_FEE = 0.01
REWARD_CLIP = 10

LEARNING_RATE=0.0025
GAMMA=0.65
EPSILON=1.0
EPSILON_DECAY=0.985
EPSILON_MIN=0.01
BUFFER_SIZE=100000
DROPOUT_RATE=0.2
WINDOW_SIZE=100
LOSS_FUNCTION = 'mean_squared_error'
PATIENCE = 10
RESTORE_BEST_WEIGHTS = True
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 0.00001
VERBOSE = 1

# %%

def scaleData(df, target):
    # Set date as index
    df.set_index('date', inplace=True)

    # Get only the data of APPLE
    df = df[df['tic'] == 'AAPL'].copy()
    df.drop(columns=['tic'], inplace=True)

    # Handling infinite values
    processed = df.replace([np.inf, -np.inf], np.nan)
    processed = processed.ffill()
    processed = processed.bfill()

    # Separating features and target
    features = processed.drop(columns=[target])
    target_values = processed[target]

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the features
    features_scaled = scaler.fit_transform(features)

    # Combine the scaled features with the target variable
    df_scaled = pd.concat(
        [pd.DataFrame(features_scaled, columns=features.columns, index=features.index), target_values], axis=1)

    return df_scaled


# Prepross data in tensorflow format so it is efficient for training
def load_data(df):
    dataset = tf.data.Dataset.from_tensor_slices((df.values[:, :-1], df.values[:, -1]))
    return dataset


def preprocess_data(dataset, batch_size=BATCH_SIZE, shuffle_buffer_size = SHUFFLE_BUFFER_SIZE):
    #shuffle datos para no introducir sesgo
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    #division de datos en lotes
    dataset = dataset.batch(batch_size)
    # Prefetching data to optimize training procesamiento de datos y la ejecución del modelo se solapen
    # el autotune permite que el tamaño del buffer se ajuste automáticamente
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


df = scaleData(data, 'close')
print(df.head())

# Prepare the dataset for training
dataset = load_data(df)
data = preprocess_data(dataset, batch_size=BATCH_SIZE)
data
# %% md
# 3. Creating the environment
# %% md
## 3.1 Create the trading environment
# %%
class CustomStockEnv(gym.Env):
    # 0 -> sell
    # 1 -> hold
    # 2 -> buy
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size= WINDOW_SIZE, transaction_fee=TRANSACTION_FEE):
        super(CustomStockEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.transaction_fee = transaction_fee
        self.shape = (window_size, len(df.columns))
        self.action_space = spaces.Discrete(3)  # Actions: 0 = sell, 1 = hold, 2 = buy
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        self.current_step = 0
        self.last_price = 0
        self.last_action = 1  # start with 'hold'

    def step(self, action):
        self.current_step += 1
        done = self.current_step + self.window_size >= len(self.df)
        current_price = self.df.loc[self.df.index[self.current_step], 'close']
        next_state = self.df.iloc[self.current_step:self.current_step + self.window_size].values

        # Calculate reward
        volatility = self.df.loc[self.df.index[max(0, self.current_step - 10):self.current_step], 'close'].std()
        reward = self.improved_reward(current_price, self.last_price, action, self.last_action, self.transaction_fee,
                                      volatility)

        # Update last price and action for next step
        self.last_price = current_price
        self.last_action = action

        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.last_price = self.df.loc[self.df.index[0], 'close']
        self.last_action = 1  # reset to 'hold'
        return self.df.iloc[self.current_step:self.current_step + self.window_size].values

    def improved_reward(self, current_close, previous_close, current_action, previous_action, transaction_fee,
                        volatility):
        # Avoid division by zero for relative price change
        if previous_close == 0:
            relative_price_change = 0
        else:
            relative_price_change = (current_close - previous_close) / previous_close

        # Logarithm argument handling to ensure it never goes to zero or negative
        log_input = relative_price_change + 1
        if log_input <= 0:
            log_return = -np.inf  # Assigning negative infinity to handle drastic drops more effectively
        else:
            log_return = np.log(log_input)

        # Risk-adjusted return handling
        if volatility == 0 or np.isnan(volatility):
            if log_return == -np.inf:
                risk_adjusted_return = -np.inf
            else:
                risk_adjusted_return = np.inf
        else:
            risk_adjusted_return = log_return / volatility

        # Transaction fee adjustment based on action
        if current_action in [0, 2]:  # apply transaction fee for buy or sell
            tf_adjustment = 1 - transaction_fee
        else:
            tf_adjustment = 1

        # Reward calculation
        if current_action in [0, 2]:  # sell or buy
            reward = risk_adjusted_return * tf_adjustment
        elif current_action == 1 and previous_action == 2:  # hold after buy -> reward if price increases, penalty if price decreases
            reward = log_return
        elif current_action == 1 and previous_action == 0:  # hold after sell -> reward if price decreases, penalty if price increases
            reward = -log_return
        else:
            reward = 0  # No action taken

        # Reward clipping to prevent outliers
        clipped_reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)

        # Log if there is an error while calculating the reward
        if np.isnan(clipped_reward):
            logging.error("NaN reward detected")
            logging.info(
                f"Current Close: {current_close}, Previous Close: {previous_close}, Action: {current_action}, Previous Action: {previous_action}")
            logging.info(
                f"Transaction Fee: {transaction_fee}, Volatility: {volatility}, Relative Price Change: {relative_price_change}")
            logging.info(
                f"Log Input: {log_input}, Log Return: {log_return}, Risk Adjusted Return: {risk_adjusted_return}, TF Adjustment: {tf_adjustment}, Reward: {reward}, Clipped Reward: {clipped_reward}")

        return clipped_reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass





def calculate_drawdown(cumulative_rewards):
    if len(cumulative_rewards) == 0:
        return 0  # No data to calculate max drawdown

    high_water_mark = []
    drawdown = []
    for value in cumulative_rewards:
        if not high_water_mark:
            high_water_mark.append(value)
        else:
            high_water_mark.append(max(high_water_mark[-1], value))

        # Prevent division by zero
        if high_water_mark[-1] == 0:
            current_drawdown = 0  # No drawdown if high water mark is zero
        else:
            current_drawdown = (high_water_mark[-1] - value) / high_water_mark[-1]

        drawdown.append(current_drawdown)

    return max(drawdown)



def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) == 0 or np.std(returns) == 0:
        return 0  # Not enough data or no variation in data to calculate Sharpe Ratio
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio


def calculate_max_drawdown(cumulative_rewards):
    if len(cumulative_rewards) == 0:
        return 0  # No data to calculate max drawdown
    high_water_mark = [cumulative_rewards[0]]  # Initialize with the first value
    drawdown = [0]  # Start with no drawdown
    for value in cumulative_rewards[1:]:
        high_water_mark.append(max(high_water_mark[-1], value))
        # Ensure the high water mark is never zero before division
        if high_water_mark[-1] != 0:
            current_drawdown = (high_water_mark[-1] - value) / high_water_mark[-1]
        else:
            current_drawdown = 0  # Avoid division by zero
        drawdown.append(current_drawdown)
    return max(drawdown)



# %% md
## 3.2 Create the agent
# %%

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=LEARNING_RATE, gamma=GAMMA,
                 epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dropout_rate=DROPOUT_RATE,window_size=WINDOW_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss_history = []  # Attribute to store loss values
        self.model = self._build_model()
        self.reward_buffer = deque(maxlen=window_size)  # Buffer to store rewards for moving average calculation

    def _build_model(self):
        model = Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))
        # First fully connected layer
        model.add(Dense(128, kernel_initializer=tf.keras.initializers.GlorotNormal()))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        model.add(ReLU())

        # Second fully connected layer
        model.add(Dense(64, kernel_initializer=tf.keras.initializers.GlorotNormal()))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        model.add(ReLU())

        # Output layer with softmax activation
        model.add(Dense(self.action_size, activation='linear', kernel_initializer=tf.keras.initializers.GlorotNormal()))
        # MSE loss function , el mas efeciente computacionalmente y el mas habitual

        # model.compile(loss=LOSS_FUNCTION, optimizer=Adam(learning_rate=self.learning_rate))
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.reward_buffer.append(reward)  # Store reward for epsilon adjustment


    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])


    # Increase or decrease epsilon based on the current reward, this is because the agent is stuck in a suboptimal policy
    # thats why the increase of the epsilon
    def update_epsilon(self):
        # Convert the deque to a list or numpy array for calculation
        if len(self.reward_buffer) > 0:
            rewards_array = np.array(self.reward_buffer)
            moving_average_reward = np.mean(rewards_array)
            current_reward = rewards_array[-1]  # The most recent reward

            # Compare current reward to the moving average
            if current_reward < moving_average_reward:
                self.epsilon = min(1.0, self.epsilon * 1.05)  # Increase epsilon if current reward is less than average
            else:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Usual decay

            # Optionally log the current epsilon value
            logging.info(f"Updated Epsilon: {self.epsilon}")
        else:
            # Handle case where there are no rewards in the buffer
            logging.info("No rewards to update epsilon.")



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
            q_next = self.model.predict(next_states_np, verbose=0)
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
            target_f = self.model.predict(np.array([np.reshape(state, (self.state_size,))]), verbose=0)
            target_f[0][action] = target
            targets_f.append(target_f[0])

            if any(np.isnan(target_f[0])):  # Check if the target function has NaNs
                logging.error("NaN detected in target values during training")

        # Perform a single batch update to the model
        history = self.model.train_on_batch(states_np, np.array(targets_f))
        self.loss_history.append(history)  # Log loss

        # Epsilon decay
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# %%
# Modified source code tensorflow.keras.callbacks.EarlyStopping corrected to restore the best weights self.best error
class CustomEarlyStopping(Callback):
    def __init__(self, monitor='loss', patience=PATIENCE, verbose=1, restore_best_weights=RESTORE_BEST_WEIGHTS, baseline=None):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if self.monitor == 'acc':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.wait = 0
        if self.baseline is not None:
            self.best = self.baseline

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if self.monitor_op(current, self.best):
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                        logging.info('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

        if self.verbose > 0:
            print(f'Epoch {epoch + 1}: early stopping patience {self.wait}/{self.patience}')

# Modified source code tensorflow.keras.callbacks.ReduceLROnPlateau to log the learning rate changes
class CustomReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

  Models often benefit from reducing the learning rate by a factor
  of 2-10 once learning stagnates. This callback monitors a
  quantity and if no improvement is seen for a 'patience' number
  of epochs, the learning rate is reduced.

  Example:

  ```python
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  model.fit(X_train, Y_train, callbacks=[reduce_lr])
  ```

  Args:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
      min_lr: lower bound on the learning rate.
  """

    def __init__(self,
                 monitor='loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(CustomReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                            'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
    """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.learning_rate)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(self.model.optimizer.learning_rate)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        backend.set_value(self.model.optimizer.learning_rate, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        logging.info('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                     'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


# Helper class to combine multiple callbacks
# This class is used to apply the EarlyStopping and ReduceLROnPlateau callbacks based on the loss value of the model
# It also allows to change the learning rate of the model when ReduceLROnPlateau is triggered
# and the earlystopping callback to stop the training when the model does not improve in 10 consecutive epochs
class CombinedCallback:
    def __init__(self, monitor='loss', early_stopping_patience=EARLY_STOPPING_PATIENCE, reduce_lr_patience=REDUCE_LR_PATIENCE,
                 reduce_lr_factor=REDUCE_LR_FACTOR, min_lr=MIN_LR, verbose=1):
        self.early_stopping = CustomEarlyStopping(monitor=monitor, patience=early_stopping_patience,
                                                  restore_best_weights=True, verbose=verbose)
        self.reduce_lr = CustomReduceLROnPlateau(monitor=monitor, factor=reduce_lr_factor,
                                                 patience=reduce_lr_patience, min_lr=min_lr, verbose=verbose)
        self.callbacks = [self.early_stopping, self.reduce_lr]
        self.model = None

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def apply_callbacks(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


# %% md
# 4. Training
# %%


class Logger:
    def __init__(self, agent, env, episodes):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.DEBUG, filename='dqn_training.log',
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log_initial_parameters(self):
        logging.info(f"Training started with {self.episodes} episodes")
        # Log parameters of the environment
        logging.info(
            f"CustomStockEnv created with the following parameters: window_size: {self.env.window_size}, transaction_fee: {self.env.transaction_fee}, shape: {self.env.shape}, action_space: {self.env.action_space}, observation_space: {self.env.observation_space}, current_step: {self.env.current_step}, last_price: {self.env.last_price}, last_action: {self.env.last_action}")
        # Log parameters of the agent
        logging.info(
            f"DQNAgent created with the following parameters: state_size: {self.agent.state_size}, action_size: {self.agent.action_size}, memory: {len(self.agent.memory)}, gamma: {self.agent.gamma}, epsilon: {self.agent.epsilon}, epsilon_min: {self.agent.epsilon_min}, epsilon_decay: {self.agent.epsilon_decay}, batch_size: {self.agent.batch_size}, learning_rate: {self.agent.learning_rate}, dropout_rate: {self.agent.dropout_rate}")



    def log_episode(self, episode, steps, total_reward, start_time, end_time):
        duration = end_time - start_time
        logging.info(
            f"Episode: {episode}/{self.episodes}, Steps: {steps}, Total Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.4f}, Duration: {duration:.2f}s")
        print(
            f"Episode: {e}/{episodes}, Steps: {steps}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Duration: {end_time - start_time:.2f}s")






def log_metrics(episode, total_reward, drawdown, sharpe_ratio, max_drawdown):
    logging.info(f"Episode: {episode}, Total Reward: {total_reward}, Drawdown: {drawdown}, Sharpe Ratio: {sharpe_ratio}, Max Drawdown: {max_drawdown}")

# %%
#
#Train loop
env = CustomStockEnv(df=df, window_size=20)
# Create the env and init the agent
state_size = env.window_size * len(df.columns)
action_size = 3  # buy, sell, hold
agent = DQNAgent(state_size=state_size, action_size=action_size)

print("Start training")
episodes = 5

combined_callback = CombinedCallback(monitor='loss', early_stopping_patience=EARLY_STOPPING_PATIENCE, reduce_lr_patience=REDUCE_LR_PATIENCE,
                                     reduce_lr_factor=REDUCE_LR_FACTOR, min_lr=MIN_LR, verbose=1)
combined_callback.set_model(agent.model)

callback_states = {
    'early_stopping_wait': combined_callback.early_stopping.wait,
    'early_stopping_best': combined_callback.early_stopping.best,
    'early_stopping_stopped_epoch': combined_callback.early_stopping.stopped_epoch,
    'reduce_lr_wait': combined_callback.reduce_lr.wait,
    'reduce_lr_best': combined_callback.reduce_lr.best,
    'reduce_lr_cooldown_counter': combined_callback.reduce_lr.cooldown_counter,
}
# Initialize the logger
logger = Logger(agent, env, episodes)
logger.log_initial_parameters()

for e in range(episodes):
    start_time = time.time()
    cumulative_rewards = []

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    steps = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        cumulative_rewards.append(reward)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            total_reward = sum(cumulative_rewards)
            drawdown = calculate_drawdown(cumulative_rewards)
            sharpe_ratio = calculate_sharpe_ratio(np.array(cumulative_rewards))
            max_drawdown = calculate_max_drawdown(cumulative_rewards)

            end_time = time.time()
            log_metrics(e, total_reward, drawdown, sharpe_ratio, max_drawdown)
            logger.log_episode(e, steps, total_reward, start_time, end_time)

            with writer.as_default():
                tf.summary.scalar('Total Reward', total_reward, step=e)
                tf.summary.scalar('Steps', steps, step=e)
                tf.summary.scalar('Epsilon', agent.epsilon, step=e)
            writer.flush()
            break

        agent.replay()
        agent.update_epsilon()

    # Checkpoint every 1 episodes
    if (e + 1) % 1 == 0:
        model_path = os.path.join(f'dqn_model_{e}.h5')
        agent.model.save(model_path)

        training_state_path = os.path.join(f'dqn_training_state_{e}.pkl')
        with open(training_state_path, 'wb') as f:
            pickle.dump({'episode': e, 'epsilon': agent.epsilon, 'callback_states': callback_states}, f)

    # Apply the combined callbacks using the loss value
    if agent.loss_history:
        with writer.as_default():
            tf.summary.scalar('Loss', agent.loss_history[-1], step=e)
        writer.flush()
        logging.info(f"Agent loss in episode {e} : {agent.loss_history[-1]}")
        combined_callback.apply_callbacks(e, logs={'loss': agent.loss_history[-1]})

    # Check if EarlyStopping has stopped the training
    if combined_callback.early_stopping.stopped_epoch > 0:
        print(f"Early stopping triggered at episode {e}")
        break

# Log final loss history
with open('loss_history.pkl', 'wb') as f:
    pickle.dump(agent.loss_history, f)

print("End training")

##%%
# %% md

# ---------------------------------------------- 5. Resuming training ----------------------------------------------

# from tensorflow.keras.models import load_model
#
#
#
# env = CustomStockEnv(df=df, window_size=20)
# # Create the env and init the agent
# state_size = env.window_size * len(df.columns)
# action_size = 3  # buy, sell, hold
# agent = DQNAgent(state_size=state_size, action_size=action_size)
# combined_callback = CombinedCallback(monitor='loss', early_stopping_patience=10, reduce_lr_patience=5,
#                                      reduce_lr_factor=0.5, min_lr=0.00001, verbose=1)
# combined_callback.set_model(agent.model)
#
#
# episodes = 100
#
# # Initialize the logger
# logger = Logger(agent, env, episodes)
# logger.log_initial_parameters()
#
# last_episode = 0
#
# # Load the model
# model_path = os.path.join(f'dqn_model_{last_episode}.h5')
# agent.model = load_model(model_path)
# agent.model.compile(optimizer=Adam(learning_rate=agent.learning_rate), loss='mean_squared_error')
#
# # Load the training state and callback states
# training_state_path = os.path.join(f'dqn_training_state_{last_episode}.pkl')
# with open(training_state_path, 'rb') as f:
#     training_state = pickle.load(f)
#
# agent.epsilon = training_state['epsilon']
# callback_states = training_state['callback_states']
#
# combined_callback.early_stopping.wait = callback_states['early_stopping_wait']
# combined_callback.early_stopping.best = callback_states['early_stopping_best']
# combined_callback.early_stopping.stopped_epoch = callback_states['early_stopping_stopped_epoch']
#
# combined_callback.reduce_lr.wait = callback_states['reduce_lr_wait']
# combined_callback.reduce_lr.best = callback_states['reduce_lr_best']
# # combined_callback.reduce_lr.last_epoch = callback_states['reduce_lr_last_epoch']
# combined_callback.reduce_lr.cooldown_counter = callback_states['reduce_lr_cooldown_counter']
#
# combined_callback.set_model(agent.model)  # Re-set the model to ensure callbacks are properly associated
#
#
#
#
# print("Start training")
# for e in range(last_episode + 1, episodes):
#     start_time = time.time()
#
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])
#     total_reward = 0
#     steps = 0
#
#     while True:
#         action = agent.choose_action(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_size])
#
#         agent.remember(state, action, reward, next_state, done)
#
#         state = next_state
#         total_reward += reward
#         steps += 1
#
#         if done:
#             end_time = time.time()
#             logger.log_episode(e, steps, total_reward, start_time, end_time)
#             break
#
#         agent.replay()
#
#     # Save model and training state every episode
#     model_path = os.path.join(f'dqn_model_{e}.h5')
#     agent.model.save(model_path)
#
#     training_state_path = os.path.join(f'dqn_training_state_{e}.pkl')
#     with open(training_state_path, 'wb') as f:
#         pickle.dump({'episode': e, 'epsilon': agent.epsilon}, f)
#
#     # Apply callbacks and check for early stopping
#     if agent.loss_history:
#         logging.info(f"Agent loss in episode {e} : {agent.loss_history[-1]}")
#         combined_callback.apply_callbacks(e, logs={'loss': agent.loss_history[-1]})
#
#     if combined_callback.early_stopping.stopped_epoch > 0:
#         print(f"Early stopping triggered at episode {e}")
#         break
#
# # Save final loss history at the end of training
# with open('loss_history.pkl', 'wb') as f:
#     pickle.dump(agent.loss_history, f)
#
# print("End of resumed training")
