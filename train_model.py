import custom_env

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import json

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = custom_env.Bisca()
np.random.seed(RANDOM_SEED)

# print("Action Space: {}".format(env.action_space))
# print("State space: {}".format(env.observation_space))

### SETUP ###
#-> model name
model_name = "bianca_v5"

# -> Neural network
learning_rate_nt = 0.001

# -> Bellmans equation
learning_rate = 0.4
discount_factor = 0.618

# -> Train
train_episodes = 20_000
steps_to_update_target_model = 100
steps_to_update_p2_model = 300
steps_to_train_model = 9

MIN_REPLAY_SIZE = 1_000
batch_size = 64 * 3

# -> Initial models
model_main = "bianca_v4"
model_p2 = "bianca_v4"

# -> rewards
rewards = {"win_round_r": 5, "win_ep_r": 0}

# -> Exploration
epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1  # You can't explore more than 100% of the time
min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time

percent_at_70 = 0.05
decay = -np.log((max_epsilon * percent_at_70 - min_epsilon) /
                (max_epsilon - min_epsilon))/(0.7 * train_episodes)
######

custom_env.Bisca.REWARDS = rewards

epsilon_setup = {"epsilon": epsilon, "max_epsilon":max_epsilon, "min_epsilon": min_epsilon, "decay": decay}

train_setup = {"train_episodes":train_episodes, "steps_to_update_target_model":steps_to_update_target_model,
                "steps_to_update_p2_model":steps_to_update_p2_model, "steps_to_train_model":steps_to_train_model,
                "MIN_REPLAY_SIZE": MIN_REPLAY_SIZE, "batch_size":batch_size, "learning_rate_nt":learning_rate_nt, 
                "learning_rate": learning_rate, "discount_factor":discount_factor, "model_main":model_main, 
                "model_p2": model_p2, "rewards": rewards}

def agent(state_shape, action_shape, learning_rate):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24*3, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(24*2, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(30, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(18, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape,activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model


def train(replay_memory, model, target_model, train_setup):
    MIN_REPLAY_SIZE = train_setup["MIN_REPLAY_SIZE"]
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    learning_rate = train_setup["learning_rate"]
    discount_factor = train_setup["discount_factor"]

    batch_size = train_setup["batch_size"]
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done, info) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index][info:])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        # min_q = current_qs.min()
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q
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
    model_main = train_setup["model_main"]
    model_p2 = train_setup["model_p2"]
    
    if model_main == None:
        model = agent(env.OBSERVATION_N, env.ACTION_N, train_setup["learning_rate_nt"])
    else:
        model = tf.keras.models.load_model(f"saved_model/{model_main}")
    
    target_model = agent(env.OBSERVATION_N, env.ACTION_N, train_setup["learning_rate_nt"])
    target_model.set_weights(model.get_weights())

    if model_p2 == None:
        p2_model = agent(env.OBSERVATION_N, env.ACTION_N, train_setup["learning_rate_nt"])
        p2_model.set_weights(model.get_weights())
    else:
        p2_model = tf.keras.models.load_model(f"saved_model/{model_p2}")
    
    # model = tf.keras.models.load_model(f"saved_model/bianca_v1")
    # target_model = tf.keras.models.load_model(f"saved_model/bianca_v1")
    # target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    update_target_model_count = 0
    update_p2_model_count = 0
    train_model_count = 0
    steps_to_update_target_model = train_setup["steps_to_update_target_model"]
    steps_to_update_p2_model = train_setup["steps_to_update_p2_model"]
    steps_to_train_model = train_setup["steps_to_train_model"]

    save_reward_freq = 200
    total_reward_list = np.zeros(int(train_episodes/save_reward_freq))
    reward_sum = 0
    reward_count = 0

    freq_prog = int(train_episodes/30)
    # save_unique_card = np.random.random(train_episodes) > 0.5
    is_first_move = np.random.random(train_episodes) > 0.5
    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset(
            first_move=is_first_move[episode], model=p2_model)

        done = False
        while not done:
            update_target_model_count += 1
            update_p2_model_count += 1
            train_model_count += 1

            random_number = random.random()
            # Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.get_action(model="Random", player=0) 
            else:
                # Exploit best known action
                action = env.get_action(model, 0)

            new_observation, reward, done, info = env.step(action, model=p2_model)

            # save_obs = True
            # if info == 3:
            #     save_obs = save_unique_card[episode]

            # if save_obs:
            #     replay_memory.append([env.reshape_state(observation), action, reward, 
            #                         env.reshape_state(new_observation), done, info])
            replay_memory.append([env.reshape_state(observation), action, reward, 
                                env.reshape_state(new_observation), done, info])

            # Update the Main Network using the Bellman Equation
            if train_model_count > steps_to_train_model:
                train_model_count =  0
                train(replay_memory, model, target_model, train_setup)

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

                if update_target_model_count > steps_to_update_target_model:
                    # print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    update_target_model_count = 0

                if update_p2_model_count > steps_to_update_p2_model:
                    p2_model.set_weights(model.get_weights())
                    update_p2_model_count = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    model.save(f'saved_model/{model_name}')
    with open(f"training_config/{model_name}.json", "w") as config_file:
        config_file.write(json.dumps([epsilon_setup, train_setup]))

    plt.scatter(np.arange(total_reward_list.size), total_reward_list)
    plt.show()

    # model_p2 = tf.keras.models.load_model(f"saved_model/bianca")
    # test_against_model(1000, model, model_p2)

star_time = time.time()
main(epsilon_setup, train_setup)
end_time = time.time()

total_time = (end_time - star_time)/60
print(f'Tempo de execução: {total_time:.2f} min')
