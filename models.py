import tensorflow as tf
keras = tf.keras

import numpy as np
import random
from collections import deque, namedtuple
import random
import time
import os
import pickle


from observation import Observation, observation_collection
import bisca_env as bisca_env 
import players as players
from debug_model import DebugModel, TestStates
import utils
from loss import compute_loss

class Exploration:
    name = "exploration"

    def __init__(self, min_epsilon: float, x_percent_at_k_min: float, train_episodes: int, k:float=2) -> None:
        self.min_epsilon = min_epsilon
        self.x_percent_at_k_min = x_percent_at_k_min
        self.k = k
        self.train_episodes = train_episodes

        self.decay = 1/(train_episodes * x_percent_at_k_min) * np.log(min_epsilon * (k -1) / (1 - min_epsilon))
    
    def get_epsilon(self, x: float):
        return self.min_epsilon + (1 - self.min_epsilon) * np.exp(self.decay * x)

    def save_obj(self):
        return {
            "min_epsilon": self.min_epsilon,
            "x_percent_at_k_min": self.x_percent_at_k_min,
            "train_episodes": self.train_episodes,
            "k": self.k,
        }

class Dnn:
    class TrainConfig:
        name = "training_cfg"

        def __init__(self, train_episodes: int, batch_size: int, steps_to_update_target_model: int,
            steps_to_update_p2_model: int,steps_to_train_model: int, min_replay_size: int) -> None:
            self.train_episodes = train_episodes
            self.batch_size= batch_size
            self.steps_to_update_target_model = steps_to_update_target_model
            self.steps_to_update_p2_model = steps_to_update_p2_model
            self.steps_to_train_model = steps_to_train_model
            self.min_replay_size = min_replay_size
        
        def save_obj(self):
            return {
                "train_episodes": self.train_episodes,
                "batch_size": self.batch_size,
                "steps_to_update_target_model": self.steps_to_update_target_model,
                "steps_to_update_p2_model": self.steps_to_update_p2_model,
                "steps_to_train_model": self.steps_to_train_model,
                "min_replay_size": self.min_replay_size,
            } 

    class BellmansEqConfig:
        name = "bellmans_cfg"

        def __init__(self, learning_rate: float, discount_factor: float) -> None:
            self.learning_rate = learning_rate
            self.discount_factor= discount_factor

        def save_obj(self):
            return {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
            }

    class NeuralNetworkConfig:
        name = "neural_network_cfg"
        def __init__(self, learning_rate: float, tau: float) -> None:
            self.learning_rate = learning_rate
            self.tau =tau
        
        def save_obj(self):
            return {
                "learning_rate": self.learning_rate,
                "tau": self.tau,
            }
    
    configs = [TrainConfig, BellmansEqConfig, NeuralNetworkConfig, Exploration]
    model_names = ["model", "p2_model", "target_model"]
    checkpoint_folder_path = {"root": "checkpoint"} 
    checkpoint_folder_path["config"] = os.path.join(checkpoint_folder_path["root"], "config")
    checkpoint_folder_path["debug"] = os.path.join(checkpoint_folder_path["root"], "debug")

    for path in checkpoint_folder_path.values():
        if not os.path.exists(path):
            os.mkdir(path)
    
    MAX_REPLAY_SIZE = 50_000

    num_cards_hand_to_mask = {0: [False, False, False], 1:[True, False, False], 2:[True, True, False], 3: [True, True, True]}
    Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "num_hand_cards"])

    def __init__(self, train_cfg: TrainConfig, bellmans_cfg: BellmansEqConfig, neural_network_cfg: NeuralNetworkConfig, 
        exploration: Exploration, observation_system: Observation, name: str, load_model_name: str = None, from_checkpoint: bool = False) -> None:
        self.from_checkpoint = from_checkpoint
        if from_checkpoint:
            self.load_checkpoint()
            return 
        
        self.train_cfg = train_cfg
        self.bellmans_cfg = bellmans_cfg
        self.neural_network_cfg= neural_network_cfg
        self.exploration = exploration
        self.inst_configs = [self.train_cfg, self.bellmans_cfg, self.neural_network_cfg, self.exploration]

        self.observation_system = observation_system
        self.name = name

        self.optimizer = tf.keras.optimizers.Adam(self.neural_network_cfg.learning_rate)
        self.replay_memory: deque[self.Experience]

        if load_model_name == None:
            self.model = self.create_agent()
        else:
            self.model = keras.models.load_model(os.path.join("saved_model", load_model_name + ".h5"))

    def save_checkpoint(self, episode: int):
        root_path = self.checkpoint_folder_path["root"]
        config_root_path = self.checkpoint_folder_path["config"]
        debug_path = self.checkpoint_folder_path["debug"]

        for ints_config in self.inst_configs:
            with open(os.path.join(config_root_path, ints_config.name + ".pickle"), "wb") as f:
                pickle.dump(ints_config.save_obj(), f)
        
        model_cfg = {
            "name": self.name, "obs_type": self.observation_system.name, "current_ep": episode,
            "update_target_model_count": self.update_target_model_count, "update_p2_model_count": self.update_p2_model_count,
            "train_model_count": self.train_model_count,
        }
        with open(os.path.join(root_path, "model_cfg.pickle"), "wb") as f:
            pickle.dump(model_cfg, f)

        replay_memory_data = []
        for r_i in self.replay_memory:
            replay_memory_data.append([r_i.state, r_i.action, r_i.reward,
                r_i.next_state, r_i.done, r_i.num_hand_cards])

        with open(os.path.join(root_path, "memory.pickle"), "wb") as f:
            pickle.dump(replay_memory_data, f)
        
        self.model.save(os.path.join(root_path, "model.h5"))
        self.p2_model.save(os.path.join(root_path, "p2_model.h5"))
        self.target_model.save(os.path.join(root_path, "target_model.h5"))

        self.debug.save_data(debug_path)
        self.test_states.save_data(debug_path)

    def load_checkpoint(self):
        root_path = self.checkpoint_folder_path["root"]
        config_root_path = self.checkpoint_folder_path["config"]

        cfg_data = {}
        for CfgClass in self.configs:
            with open(os.path.join(config_root_path, CfgClass.name + ".pickle"), "rb") as f:
                cfg_data[CfgClass.name] = pickle.load(f)

        self.train_cfg = self.TrainConfig(**cfg_data[self.TrainConfig.name])
        self.bellmans_cfg = self.BellmansEqConfig(**cfg_data[self.BellmansEqConfig.name])
        self.neural_network_cfg= self.NeuralNetworkConfig(**cfg_data[self.NeuralNetworkConfig.name])
        self.exploration = Exploration(**cfg_data[Exploration.name])
        self.inst_configs = [self.train_cfg, self.bellmans_cfg, self.neural_network_cfg, self.exploration]
        
        self.optimizer = tf.keras.optimizers.Adam(self.neural_network_cfg.learning_rate)

        with open(os.path.join(root_path, "model_cfg.pickle"), "rb") as f:
            model_cfg = pickle.load(f)
            self.name = model_cfg["name"]
            self.observation_system = observation_collection[model_cfg["obs_type"]]()
            self.current_episode = model_cfg["current_ep"]
            self.update_target_model_count = model_cfg["update_target_model_count"]
            self.update_p2_model_count = model_cfg["update_p2_model_count"]
            self.train_model_count = model_cfg["train_model_count"]
        

        with open(os.path.join(root_path, "memory.pickle"), "rb") as f:
            replay_memory_data = pickle.load(f)
            self.replay_memory = deque(maxlen=self.MAX_REPLAY_SIZE)
            for i in replay_memory_data:
                self.replay_memory.append(self.Experience(*i))

        models = {}
        for model_name in self.model_names:
            models[model_name] = keras.models.load_model(os.path.join(root_path, model_name))
        
        self.model = models["model"]
        self.p2_model = models["p2_model"]
        self.target_model = models["target_model"]

    def create_agent(self) -> keras.Sequential:
        """ The agent maps X-states to Y-actions
        e.g. The neural network output is [.1, .7, .1, .3]
        The highest value 0.7 is the Q-Value.
        The index of the highest action (0.7) is action #1.
        """
        state_shape = self.observation_system.OBSERVATION_N
        action_shape = self.observation_system.ACTION_N

        init = tf.keras.initializers.HeUniform(23012003)
        # model = keras.models.Sequential([
        #     keras.layers.Dense(24*3, input_shape=state_shape, activation='relu', kernel_initializer=init),
        #     keras.layers.Dense(24*2, activation='relu', kernel_initializer=init),
        #     keras.layers.Dense(30, activation='relu', kernel_initializer=init),
        #     keras.layers.Dense(18, activation='relu', kernel_initializer=init),
        #     keras.layers.Dense(8, activation='relu', kernel_initializer=init),
        #     keras.layers.Dense(action_shape,activation='linear', kernel_initializer=init),
        # ])
        model = keras.models.Sequential([
            keras.layers.Dense(30, input_shape=state_shape, activation='relu', kernel_initializer=init),
            keras.layers.Dense(30, activation='relu', kernel_initializer=init),
            keras.layers.Dense(15, activation='relu', kernel_initializer=init),
            keras.layers.Dense(15, activation='relu', kernel_initializer=init),
            keras.layers.Dense(8, activation='relu', kernel_initializer=init),
            keras.layers.Dense(action_shape,activation='linear', kernel_initializer=init,),
        ])
        
        learning_rate = self.neural_network_cfg.learning_rate
        # loss = tf.keras.losses.Huber()
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                            metrics=['accuracy'])
    
        return model

    def get_experiences(self, mini_batch_size: int):
        """
        Returns a random sample of experience tuples drawn from the memory buffer.

        Retrieves a random sample of experience tuples from the given memory_buffer and
        returns them as TensorFlow Tensors. The size of the random sample is determined by
        the mini-batch size (MINIBATCH_SIZE). 
        
        Args:
            memory_buffer (deque):
                A deque containing experiences. The experiences are stored in the memory
                buffer as namedtuples: namedtuple("Experience", field_names=["state",
                "action", "reward", "next_state", "done"]).

        Returns:
            A tuple (states, actions, rewards, next_states, done_vals) where:

                - states are the starting states of the agent.
                - actions are the actions taken by the agent from the starting states.
                - rewards are the rewards received by the agent after taking the actions.
                - next_states are the new states of the agent after taking the actions.
                - done_vals are the boolean values indicating if the episode ended.

            All tuple elements are TensorFlow Tensors whose shape is determined by the
            mini-batch size and the given Gym environment. For the Lunar Lander environment
            the states and next_states will have a shape of [MINIBATCH_SIZE, 8] while the
            actions, rewards, and done_vals will have a shape of [MINIBATCH_SIZE]. All
            TensorFlow Tensors have elements with dtype=tf.float32.
        """
        experiences = random.sample(self.replay_memory, k=mini_batch_size)
        states = tf.convert_to_tensor(
            np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
        )
        actions = tf.convert_to_tensor(
            np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
        )
        rewards = tf.convert_to_tensor(
            np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
        )
        next_states = tf.convert_to_tensor(
            np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
        )
        done_vals = tf.convert_to_tensor(
            np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
            dtype=tf.float32,
        )
        mask = tf.convert_to_tensor(
            np.array([self.num_cards_hand_to_mask[e.num_hand_cards] for e in experiences]),
            dtype=tf.bool,
        )
        return (states, actions, rewards, next_states, done_vals, mask)

    def train_neural_network_old(self):
        MIN_REPLAY_SIZE = self.train_cfg.min_replay_size
        if len(self.replay_memory) < MIN_REPLAY_SIZE:
            return
        
        learning_rate = self.bellmans_cfg.learning_rate
        discount_factor = self.bellmans_cfg.discount_factor

        batch_size = self.train_cfg.batch_size
        states, actions, rewards, next_states, done_vals, num_hand_cards = self.get_experiences(batch_size)



        mini_batch = random.sample(self.replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states, verbose=0)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done, num_cards) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index][:num_cards])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            # min_q = current_qs.min()
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q
            # current_qs[np.invert(info[0])] = min_q

            X.append(observation)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    def train_neural_network(self):
        if len(self.replay_memory) < self.train_cfg.min_replay_size:
            return

        self.agent_learn(self.get_experiences(self.train_cfg.batch_size), self.bellmans_cfg.discount_factor)

    @tf.function
    def agent_learn(self, experiences: tuple, gamma: float):
        """
        Updates the weights of the Q networks.
        
        Args:
        experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
        gamma: (float) The discount factor.
        
        """
        # Calculate the loss
        with tf.GradientTape() as tape:
            loss = compute_loss(experiences, gamma, self.model , self.target_model)

        # Get the gradients of the loss with respect to the weights.
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Update the weights of the q_network.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # update the weights of target q_network
        TAU = self.neural_network_cfg.tau
        for target_weights, q_net_weights in zip(self.target_model.weights, self.model.weights):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

    def train(self):
        if not self.from_checkpoint:
            ### Initialize the models ###
            self.target_model = self.create_agent()
            self.target_model.set_weights(self.model.get_weights())

            self.p2_model = self.create_agent()
            self.p2_model.set_weights(self.model.get_weights())
            ######

            self.current_episode = 0

            self.replay_memory = deque(maxlen=self.MAX_REPLAY_SIZE)

            self.update_target_model_count = 0
            self.update_p2_model_count = 0
            self.train_model_count = 0

        ### Initialize bisca environment ###
        player1 = players.NeuralNetwork(self.model, self.observation_system)
        # player2 = players.NeuralNetwork(self.p2_model, self.observation_system)
        player2 = players.MarcosStrat()
        self.env = bisca_env.Bisca(player2, self.observation_system)
        ######

        train_episodes = self.train_cfg.train_episodes
        steps_to_update_target_model = self.train_cfg.steps_to_update_target_model
        steps_to_update_p2_model = self.train_cfg.steps_to_update_p2_model
        steps_to_train_model = self.train_cfg.steps_to_train_model
        
        freq_test_estate = int(train_episodes/100)
        freq_prog = int(train_episodes/30)

        ### Test State ###
        states = [
        {
            "mao": "3-c, 5-c, 5-p",
            "mesa": "v", 
            "bisca": "12-p"
        },
        {
            "mao": "10-p, 8-p, 3-o",
            "mesa": "9-p", 
            "bisca": "4-o"
        },
        # {
        #     "mao": "3-e, 12-p, 4-c",
        #     "mesa": "3-c", 
        #     "bisca": "5-p"
        # },
        {
            "mao": "10-p, 11-c, 3-o",
            "mesa": "4-p", 
            "bisca": "4-c"
        }]
        self.test_states = TestStates(self.model, player1.observation_system, states)
        if self.from_checkpoint:
            self.test_states.load_data(self.checkpoint_folder_path["debug"])
        ###

        ### Play against old model ###
        # debug_p2_model = keras.models.load_model("saved_model/simple4")
        # debug_p2 = players.NeuralNetwork(debug_p2_model, Observation()) 
        debug_p2 = players.MarcosStrat(Observation()) 
        # debug_p2 = players.Random()
        self.debug = DebugModel(player1, debug_p2, 15)
        if self.from_checkpoint:
            self.debug.load_data(self.checkpoint_folder_path["debug"])
        ###

        progress = utils.Progress(freq_prog, train_episodes)

        is_first_move = np.random.random(train_episodes) > 0.5
        for episode in range(self.current_episode, train_episodes):
            if progress.get_checkpoint_elapsed_time() > 60:
                progress.set_checkpoint_start_time()
                self.save_checkpoint(episode)

            total_training_rewards = 0
            observation = self.env.reset(play_first=is_first_move[episode])

            done = False
            epsilon = self.exploration.get_epsilon(episode)
            while not done:
                self.update_target_model_count += 1
                self.update_p2_model_count += 1
                self.train_model_count += 1

                random_number = random.random()
                if random_number <= epsilon:
                    action = player1.random_action(observation) 
                else:
                    action = player1.get_action(observation)

                new_observation, reward, done, info = self.env.step(action)

                self.replay_memory.append(self.Experience(
                    observation[0], action, reward, new_observation[0], 
                    done, info["num_hand_cards"],
                ))

                # Update the Main Network using the Bellman Equation
                if self.train_model_count > steps_to_train_model:
                    self.train_model_count =  0
                    self.train_neural_network()
                
                # if self.update_target_model_count > steps_to_update_target_model:
                #     # print('Copying main network weights to the target network weights')
                #     # self.target_model.set_weights(self.model.get_weights())
                #     # self.update_target_model_count = 0

                observation = new_observation.copy()
                total_training_rewards += reward

                if done:
                    if episode % freq_test_estate == 0:
                        self.debug.update_play_against(episode)
                        print(f"Vitória contra p2: {self.debug.pc_victory[-1]:.2f} %")
                        self.test_states.update_predictions(episode)
                        
                    progress.print(episode)

                    if self.update_p2_model_count > steps_to_update_p2_model:
                        self.p2_model.set_weights(self.model.get_weights())
                        self.update_p2_model_count = 0
                    break

        total_time = utils.format_time(time.time() - progress.start_time)
        print(f'Tempo de execução: {total_time}')

        self.model.save(f'saved_model/{self.name}.h5')

        if not os.path.exists("debug_info"):
            os.mkdir("debug_info")

        self.debug.plot_play_against()
        self.debug.save_data("debug_info")
        self.test_states.save_data("debug_info")
        self.debug.show_plots()