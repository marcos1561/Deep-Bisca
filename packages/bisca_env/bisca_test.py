import unittest
import numpy as np

from bisca_env.bisca_env import Bisca 
from bisca_env.observation import Observation, History
from bisca_env.bisca_components import Card, Suit, NullCard, Deck
import bisca_env.players as players
from bisca_env.player_hand import Hand 
import bisca_env.rewards as rewards

class Test_TestWonCard(unittest.TestCase):
    def test_won_card(self):
        c1 = Card(4, Suit.copas)
        c2 = Card(10, Suit.copas)
        bisca = Card(8, Suit.ouro)

        self.assertEqual(Bisca.winner(c1, c2, bisca), 0)
        
        c1 = Card(4, Suit.copas)
        c2 = Card(3, Suit.ouro)
        bisca = Card(8, Suit.ouro)
        self.assertEqual(Bisca.winner(c1, c2, bisca), 0)
        
        c1 = Card(2, Suit.copas)
        c2 = Card(12, Suit.espadas)
        bisca = Card(8, Suit.ouro)
        self.assertEqual(Bisca.winner(c1, c2, bisca), 1)
        
        c1 = Card(12, Suit.espadas)
        c2 = Card(3, Suit.espadas)
        bisca = Card(8, Suit.ouro)
        self.assertEqual(Bisca.winner(c1, c2, bisca), 0)

class Test_TestGame(unittest.TestCase):
    def test_game(self):
        player = players.RandomOut(History())
        env = Bisca(players.Random(), player.observation_system)
        
        for _ in range(2):
            obs = env.reset()

            done = False
            while not done:
                action = player.get_action(obs)
                new_obs, reward, done, _ = env.step(action)
                obs = new_obs

            self.assertEqual(env.scores["p1"] + env.scores["p2"], 120)
            self.assertEqual(env.player1.has_empty_hand(), True)
            self.assertEqual(env.player2.has_empty_hand(), True)

            unique_cards_history = True
            for id in range(24): 
                c = env.history.p1_cards[id]
                if c in env.history.p1_cards[:id] or c in env.history.p1_cards[id+1:]:
                    unique_cards_history = False
                    break
                
                c = env.history.p1_cards[id]
                if c in env.history.p2_cards[:id] or c in env.history.p2_cards[id+1:]:
                    unique_cards_history = False
                    break

            only_non_null = not (NullCard() in (env.history.p1_cards + env.history.p2_cards))

            self.assertEqual(unique_cards_history, True)
            self.assertEqual(only_non_null, True)

class Test_TestObservation(unittest.TestCase):
    def test_sort_hand(self):
        cards = [Card(12, Suit.copas), Card(3, Suit.copas), Card(3, Suit.ouro)]
        hand = Hand(cards)
        Observation.sort_hand(hand)

        correct_cards = [
            Card(3, Suit.ouro),
            Card(3, Suit.copas), 
            Card(12, Suit.copas), 
            ]
        correct_hand = Hand(correct_cards)

        sorted(hand.available_cards)
        sorted(correct_hand.available_cards)

        self.assertEqual(hand.cards, correct_hand.cards)
        self.assertEqual(hand.available_cards, correct_hand.available_cards)
        self.assertEqual(hand.empty_spaces, correct_hand.empty_spaces)

    def test_get_observation(self):
        hand = [Card(3, Suit.espadas), Card(10, Suit.espadas), Card(12, Suit.copas)]
        bisca = Card(6, Suit.copas)
        table = Card(9, Suit.ouro)

        expected = np.array([ 
            1, 0, 0, 0, 10/11,
            0, 0, 1, 0, 9/11,
            1, 0, 0, 0, 7/11,
            0, 0, 1, 0, 3/11,
            0, 1, 0, 0, 6/11,
        ])

        obs_system = Observation()
        obs = obs_system.get_observations(hand=hand, table=table, bisca=bisca, history=[]) 
        is_equal = (obs == expected).all()

        self.assertEqual(is_equal, True)
    
    def test_get_human_card(self):
        obs = np.array([[ 
            1, 0, 0, 0, 10/11,
            0, 0, 1, 0, 9/11,
            1, 0, 0, 0, 7/11,
            0, 0, 1, 0, 3/11,
            0, 1, 0, 0, 6/11,
        ]])

        hand = [Card(3, Suit.espadas), Card(10, Suit.espadas), Card(12, Suit.copas)]
        Observation.sort_hand(hand)
        bisca = Card(6, Suit.copas)
        table = Card(9, Suit.ouro)

        hand2, bisca2, table2 = Observation.get_human_cards(obs)

        self.assertEqual(hand, hand2)
        self.assertEqual(bisca, bisca2)
        self.assertEqual(table, table2)

class Test_TestObservationHistory(unittest.TestCase):
    def test(self):
        p1 = players.FirstCard(History())
        p2 = players.FirstCard(History())

        env = Bisca(p2, p1.obs_system)
        cards_order = list(range(Deck.num_cards))
        obs = env.reset(play_first=True, cards_order=cards_order)

        base_obs = Observation()

        obs_test = History().base_observation.copy()
        
        base_obs_test = base_obs.get_observations(env.player1.hand, env.table, env.bisca)
        is_obs_equal = (base_obs_test[0] == obs[0][:History.init_history_id]).all()
        self.assertEqual(is_obs_equal, True)

        h1 = Hand(Deck.cards[:3])
        h2 = Hand(Deck.cards[3:6])

        p1.obs_system.sort_hand(h1)
        p1.obs_system.sort_hand(h2)

        if h1[0].points > 0:
            obs_test[5*5 + History.cards_history_ids[h1[0].name]] = 1
        if h2[0].points > 0:
            obs_test[5*5 + History.cards_history_ids[h2[0].name]] = 1

        p1_id = 7
        p2_id = 6
        if env.winner(h1[0], h2[2], Deck.cards[-1]) == 1:
            p1_id = 6
            p2_id = 7

        h1[0] = Deck.cards[p1_id]
        h2[0] = Deck.cards[p2_id]

        p1.obs_system.sort_hand(h1)
        p1.obs_system.sort_hand(h2)
        
        if h1[0].points > 0:
            obs_test[5*5 + History.cards_history_ids[h1[0].name]] = 1
        if h2[0].points > 0:
            obs_test[5*5 + History.cards_history_ids[h2[0].name]] = 1

        for i in range(2):
            action = p1.get_action(obs)
            obs, reward, done, info = env.step(action)

            base_obs_test = base_obs.get_observations(env.player1.hand, env.table, env.bisca)
            is_obs_equal = (base_obs_test[0] == obs[0][:History.init_history_id]).all()
            self.assertEqual(is_obs_equal, True)

        equal_hist = (obs[0][History.init_history_id:] == obs_test[25:]).all()
        self.assertEqual(equal_hist, True)

class Test_TestReward(unittest.TestCase):
    def test_card_points(self):
        done = False
        scores = {"p1": None, "p2": None}
        
        p1_card = Card(3, Suit.copas)
        p2_card = Card(10, Suit.copas)
        has_won_round = True

        round_reward = (10 + 2) * 1/2

        self.assertEqual(
            rewards.CardPoints.get_reward(p1_card, p2_card, has_won_round, done, scores),
            round_reward)
        
        done = True
        scores = {"p1": 64, "p2": 56}
        p1_card = Card(1, Suit.copas)
        p2_card = Card(11, Suit.copas)
        has_won_round = True

        final_reward = (11 + 3) * 1/2 + rewards.CardPoints.win_reward

        self.assertEqual(
            rewards.CardPoints.get_reward(p1_card, p2_card, has_won_round, done, scores),
            final_reward)

    def test_bisca_punishment(self):
        done = False
        scores = {"p1": None, "p2": None}
        
        p1_card = Card(3, Suit.copas)
        p2_card = Card(2, Suit.paus)
        bisca = Card(10, Suit.copas)
        has_won_round = True

        bisca_punishment = rewards.BiscaPunishment.factor_bisca_punishment * (10) + rewards.BiscaPunishment.const_bisca_punishment
        round_reward = 10 * 1/2

        self.assertEqual(
            rewards.BiscaPunishment.get_reward(p1_card, p2_card, has_won_round, done, scores, bisca),
            round_reward - bisca_punishment)

        p1_card = Card(3, Suit.copas)
        p2_card = Card(2, Suit.copas)
        bisca = Card(10, Suit.copas)
        has_won_round = True

        bisca_punishment = 0
        round_reward = 10 * 1/2

        self.assertEqual(
            rewards.BiscaPunishment.get_reward(p1_card, p2_card, has_won_round, done, scores, bisca),
            round_reward - bisca_punishment)

        