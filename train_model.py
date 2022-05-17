# import gym
import custom_env
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import random
import matplotlib.pyplot as plt

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = custom_env.Bisca()
np.random.seed(RANDOM_SEED)

# print("Action Space: {}".format(env.action_space))
# print("State space: {}".format(env.observation_space))

### SETUP ###
#-> model name
model_name = "bianca_teste_novo"

# -> Neural network
learning_rate_nt = 0.001

# -> Bellmans equation
learning_rate = 0.1
discount_factor = 0.518

# -> Train
train_episodes = 20000
MIN_REPLAY_SIZE = 1000
batch_size = 64 * 4
# train_episodes = 200
# MIN_REPLAY_SIZE = 70
# batch_size = 64

# -> Exploration
epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1  # You can't explore more than 100% of the time
min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time

percent_at_70 = 0.05
decay = -np.log((max_epsilon * percent_at_70 - min_epsilon) /
                (max_epsilon - min_epsilon))/(0.7 * train_episodes)
######

epsilon_setup = {"epsilon": epsilon, "max_epsilon":max_epsilon, "min_epsilon": min_epsilon, "decay": decay}

train_setup = {"train_episodes":train_episodes, "MIN_REPLAY_SIZE": MIN_REPLAY_SIZE, "batch_size":batch_size, "learning_rate_nt":learning_rate_nt, "learning_rate": learning_rate, "discount_factor":discount_factor }

def agent(state_shape, action_shape, learning_rate):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24*3, input_shape=state_shape,
              activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(
        24*2, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(
        30, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(
        30, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(
        20, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(
        12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape,
              activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(
        lr=learning_rate), metrics=['accuracy'])
    return model


def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]


def train(env, replay_memory, model, target_model, train_setup):
    learning_rate = train_setup["learning_rate"]
    discount_factor = train_setup["discount_factor"]

    MIN_REPLAY_SIZE = train_setup["MIN_REPLAY_SIZE"]
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = train_setup["batch_size"]
    # batch_size = 64
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done, info) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * \
                np.max(future_qs_list[index][info:])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        # min_q = current_qs.min()
        current_qs[action] = (1 - learning_rate) * \
            current_qs[action] + learning_rate * max_future_q
        # current_qs[np.invert(info[0])] = min_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def main(epsilon_setup, train_setup):
    epsilon = epsilon_setup["epsilon"]  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = epsilon_setup["max_epsilon"]  # You can't explore more than 100% of the time
    min_epsilon = epsilon_setup["min_epsilon"]  # At a minimum, we'll always explore 1% of the time
    decay = epsilon_setup["decay"]

    # Initialize the Target and Main models
    model = agent(env.OBSERVATION_N, env.ACTION_N, train_setup["learning_rate_nt"])
    target_model = agent(env.OBSERVATION_N, env.ACTION_N, train_setup["learning_rate_nt"])
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    steps_to_update_target_model = 0

    save_reward_freq = 200
    total_reward_list = np.zeros(int(train_episodes/save_reward_freq))
    reward_sum = 0
    reward_count = 0

    freq_prog = int(train_episodes/30)
    save_unique_card = np.random.random(train_episodes) > 0.5
    is_first_move = np.random.random(train_episodes) > 0.5
    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset(
            first_move=is_first_move[episode], model=target_model)

        done = False
        while not done:
            steps_to_update_target_model += 1

            random_number = np.random.rand()
            # Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.get_action(model="Random", player=0) 
            else:
                # Exploit best known action
                action = env.get_action(model, 0)

            new_observation, reward, done, info = env.step(action, model=target_model)

            save_obs = True
            if info == 3:
                save_obs = save_unique_card[episode]

            if save_obs:
                replay_memory.append([env.reshape_state(observation), action, reward, 
                                    env.reshape_state(new_observation), done, info])

            # Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 9 == 0:
                train(env, replay_memory, model, target_model, train_setup)

            observation = new_observation
            total_training_rewards += reward

            if done:
                reward_sum += total_training_rewards
                if (episode+1) % save_reward_freq == 0:
                    total_reward_list[reward_count] = reward_sum/save_reward_freq
                    reward_count += 1
                    reward_sum = 0

                # print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                if episode % freq_prog == 0:
                    print(f"Progress: {episode/train_episodes*100:.2f} %")
                    print(f"reward_mean: {total_reward_list[reward_count-1]}")
                    # print("epsilon", epsilon)
                    # print("len_memory", len(replay_memory))

                if steps_to_update_target_model >= 100:
                    # print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    plt.scatter(np.arange(total_reward_list.size), total_reward_list)
    plt.show()

    model.save(f'saved_model/{model_name}')
    model_p2 = tf.keras.models.load_model(f"saved_model/bianca")
    # test_against_model(1000, model, model_p2)

main(epsilon_setup, train_setup)
