import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import random

import players as players
import bisca_env as env
from observation import Observation
from bisca_components import Deck

class TestStates:
    root_folder = "teste_state" 
    preds_path = "teste_state//preds.npy"
    eps_path = "teste_state//eps.npy"
    states_info_path = "teste_state//states_info.pickle"

    data_frame_folder_path = "teste_state//dfs"

    def __init__(self, model, obs_system: Observation, states: list[dict[str,str]] = []) -> None:
        self.model = model
        self.states = states

        self.observation_system = obs_system

        self.internal_states = []
        self.order_hand = []
        for str_state in self.states:
            obs, order_hand = self.observation_system.observation_from_str(str_state)
            self.internal_states.append(obs)
            self.order_hand.append(order_hand)

        self.pred_list = []
        self.ep = []
    
    def calc_predictions(self):
        preds = []
        for state in self.internal_states:
            preds.append(self.model.predict(state, verbose=0).flatten())

        return np.array(preds)

    def update_predictions(self, episode):
        self.pred_list.append(self.calc_predictions())
        self.ep.append(episode)

    def print_pred(self):
        preds = self.calc_predictions()
        for s, h, p in zip(self.states, self.order_hand, preds):
            print(s)
            for c, q in zip(h, p):
                c: str
                print(f"{c.ljust(11)}: {q:.3f}")
            print()
    
    def load_data(self, folder):
        self.pred_list = list(np.load(os.path.join(folder, self.preds_path)))
        self.ep = list(np.load(os.path.join(folder, self.eps_path)))
        
        with open(os.path.join(folder, self.states_info_path), "rb") as f:
            data = pickle.load(f)
            self.states = data["states"]
            self.order_hand = data["order_hand"] 

    def save_data(self, folder):
        root_path = os.path.join(folder, self.root_folder)
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        
        np.save(os.path.join(folder, self.preds_path), np.array(self.pred_list))
        np.save(os.path.join(folder, self.eps_path), np.array(self.ep))

        with open(os.path.join(folder, self.states_info_path), "wb") as f:
            pickle.dump({"states": self.states, "order_hand": self.order_hand}, f)
        
        main_df_folder_path = os.path.join(folder, self.data_frame_folder_path)
        if not os.path.exists(main_df_folder_path):
            os.mkdir(main_df_folder_path)
        
        for i in range(len(self.states)):
            pd.DataFrame(np.array(self.pred_list)[:,i,:], columns=self.order_hand[i], index=self.ep).to_csv(
                os.path.join(main_df_folder_path, "state" + str(i) + ".csv")
            )

class DebugModel:
    def __init__(self, player: players.NeuralNetwork, play_against_player: players.Player,
        play_against_total_eps: int) -> None:
        self.player = player

        self.play_against_total_eps = play_against_total_eps
        self.play_against_player = play_against_player
        self.pc_victory = []
        self.pc_victory_ep = []
        self.reward = []

        self.cards_order = []
        cards_ids = list(range(Deck.num_total_cards))
        for i in range(play_against_total_eps): 
            random.shuffle(cards_ids)
            self.cards_order.append(cards_ids.copy())
    
    @staticmethod
    def player_battle(player1: players.OutPLayer, player2: players.Player, total_eps: int,
            cards_ids:list=None, verbose=False):
        bisca_env = env.Bisca(player2, player1.observation_system)

        # Random bool list for randomize who plays first
        # first_move_list = np.random.random(total_eps) > 0.5
        first_move_list = np.zeros(shape=total_eps)
        first_move_list[:total_eps//2] = 1

        # Frequency to show progress
        freq_prog = int(total_eps/10)

        total_reward = 0
        wins = 0
        num_ep = 0
        while num_ep < total_eps:
            num_ep += 1
            if verbose:
                if num_ep % freq_prog == 0:
                    print(f"Progresso: {num_ep/total_eps*100:.0f} % | pc_win = {wins/num_ep*100:.2f}")

            c_ids = None
            if cards_ids != None:
                c_ids = cards_ids[num_ep-1]

            observation = bisca_env.reset(play_first=first_move_list[num_ep-1], cards_order=c_ids)
            done = False
            while not done:
                action = player1.get_action(observation)
                observation, reward, done, info = bisca_env.step(action)

                total_reward += reward
                if done:
                    if info["scores"]["p1"] > info["scores"]["p2"]:
                        wins += 1
        
        mean_reward = total_reward/total_eps
        pc_victory = wins/total_eps*100
        if verbose:
            print(f"Porcentagem de vitória: {pc_victory:.2f}")
            print(f"Reward médio          : {mean_reward:.2f}")
        
        return pc_victory, mean_reward

    def update_play_against(self, current_ep: int, verbose=False):
        pc_victory, mean_reward = self.player_battle(self.player, self.play_against_player, self.play_against_total_eps, self.cards_order, verbose)
        self.pc_victory.append(pc_victory)
        self.reward.append(mean_reward)
        self.pc_victory_ep.append(current_ep)
        
        return pc_victory

    def plot_play_against(self):
        fig, ax = plt.subplots()
        ax.plot(self.pc_victory_ep, self.pc_victory)

    def save_data(self, folder: str):
        np.save(os.path.join(folder, "pc_victory.npy"), np.array([self.pc_victory_ep, self.pc_victory, self.reward]))

    def load_data(self, folder):
        data = np.load(os.path.join(folder, "pc_victory.npy"))
        self.pc_victory_ep = list(data[0])
        self.pc_victory = list(data[1])
        self.reward = list(data[2])

    def show_plots(self):
        plt.show()

class DebugModelOld:
    # states_test = ["VAZIO- 5-e 1-e 8-p 1-c",
    #                 "9-p 5-e 1-e 10-p 2-c",
    #                 "VAZIO- 5-e VAZIO- 7-o 10-c 1-c 3-c 12-c"]

    # states_test_m = env.Bisca.human_to_machine(states_test) 
    # states_test_m_rs = []
    # for s in states_test_m:
    #     states_test_m_rs.append(env.Bisca.reshape_state(s))
    # states_test_m_rs = np.array(states_test_m_rs)
    
    def __init__(self) -> None:
        self.predictions = []
        self.pred_ep = []

        self.pc_victory_ep = []
        self.pc_victory = []

    def test_state(self, model, episode):
        ''' 
            Make the model give prediction for all the states in the static property 'states_m_rs'
            and store the result in the variable 'self.predictions'
        '''
        
        predicted = model.predict(DebugModel.states_test_m_rs, verbose=0)
        self.predictions.append(predicted)
        self.pred_ep.append(episode)

    def plot_predictions(self):
        '''
            Make a plot of the predictions stored in 'self.predictions'. 
            Each state has it's own figure and the plot consist in the values for each actions against
            the episode it was calculated.
        '''
        # Show the states in the console
        for id, s in enumerate(DebugModel.states_test_m):
            print(f"Estado {id}:")
            env.Bisca.print_state(s)
            print()

        # plot the predictions
        self.predictions = np.array(self.predictions)
        for s_id in range(len(DebugModel.states_test)):
            fig, ax = plt.subplots()
            ax.set_title(f"Estado {s_id}")

            s_p = self.predictions[:,s_id,:]
            for i in range(3):
                ax.plot(self.pred_ep, s_p[:, i], "-o", label=f"Predição {i}")
            ax.legend()
        # plt.show()

    def test_play_against(self, ep, total_eps, model1="Random", model2="Random", verbose=False):
        '''
        Test the model against other model. This function will print
        tha victory percentage of model1.

        Parameters:
            total_eps: int
                Number of episodes the models will play.

            model1: tf model
                player 1 model

            model1: tf model
                player 2 model
        '''
        bisca_env = env.Bisca()

        wins = 0
        num_ep = 0

        # Random bool list for randomize who plays first
        first_move_list = np.random.random(total_eps) > 0.5
        # Frequency to show progress
        freq_prog = int(total_eps/10)
        while num_ep < total_eps:
            num_ep += 1
            if verbose:
                if num_ep % freq_prog == 0:
                    print(f"Progresso: {num_ep/total_eps*100:.0f} % | pc_win = {wins/num_ep*100:.2f}")

            observation = bisca_env.reset(first_move=first_move_list[num_ep-1], model=model2)
            done = False
            while not done:
                action = bisca_env.get_action(model1, player=0)
                new_state, reward, done, info = bisca_env.step(action, model=model2)

                if done:
                    if bisca_env.player_points[0] > bisca_env.player_points[1]:
                        wins += 1

        pc_victory = wins/total_eps*100
        if verbose:
            print(f"Porcentam de vitória: {pc_victory:.2f}")

        
        self.pc_victory.append(pc_victory)
        self.pc_victory_ep.append(ep)
        return pc_victory


    def plot_play_against(self):
        fig, ax = plt.subplots()
        ax.plot(self.pc_victory_ep, self.pc_victory)
    
    @staticmethod
    def plot_show():
        plt.show()

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    
    model = load_model("saved_model/teste3")
    
    states = [
        {
            "mao": "10-e, 3-p, 5-c",
            "mesa": "7-p", 
            "bisca": "6-e"
        },
        {
            "mao": "10-e, 10-p, 5-c",
            "mesa": "v", 
            "bisca": "6-e"
        },
        {
            "mao": "8-p, 5-c, 8-e",
            "mesa": "2-o", 
            "bisca": "4-c"
        },
    ]

    t = TestStates(model, states)
    for i in range(3):
        t.update_predictions(i)

    print(t.pred_list)

    # from players import Random, RandomOut, NeuralNetwork
    # from observation import Observation

    # model1 = load_model("saved_model/teste3")
    # p1 = NeuralNetwork(model1, Observation)
    
    # model2 = load_model("saved_model/teste2")
    # p2 = NeuralNetwork(model2, Observation)
    # # p2 = Random()

    # DebugModel.player_battle(p1, p2, 20, verbose=True)

    # p1 = RandomOut(Observation)

    # d = DebugModel(p1, p2, 40)

    # for i in range(1):
    #     print(i)
    #     d.update_play_against(i)

    # print("media:", np.array(d.pc_victory).mean())
    # print("std:", np.array(d.pc_victory).std())
    
    # plt.hist(d.pc_victory)
    # d.plot_play_against()
    # d.save_date()
    # d.show_plots()

    # DebugModel.player_battle(RandomOut(Observation), Random(), total_eps=10000, verbose=True)

    # num_sim = 10
    # pc_array = np.zeros(num_sim)
    # for i in range(num_sim):
    #     debug.test_play_against(i, 40)


    # debug.plot_play_against()
    # debug.plot_show()
    # print("mean:", np.array(debug.pc_victory).mean())
    # print("std:", np.array(debug.pc_victory).std())
    # plt.hist(debug.pc_victory, bins=40)
    # plt.show()

    # debug_model = DebugModel()

    # model = tf.keras.models.load_model("saved_model/" + "bianca_v3")
    
    # debug_model.test_state(model, 1)
    # debug_model.test_state(model, 10)

    # debug_model.plot_predictions()
