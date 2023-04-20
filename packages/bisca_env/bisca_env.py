import numpy as np

from bisca_env.bisca_components import Card, Deck, NullCard
from bisca_env.players import Player, EnvPlayer
from bisca_env.player_hand import Hand
from bisca_env.observation import Observation
import bisca_env.rewards as rewards

class History:
    def __init__(self, observation_system: Observation, p2_obs_system: Observation = None) -> None:
        '''
        History of cards played in a game.

        The first element in `self.p1_cards` is the first card played and so on.
        '''
        self.half_total_cards = int(Deck.num_total_cards/2)

        self.p1_cards: list[Card]
        self.p2_cards: list[Card]
        self.p1_first: list[bool]
        self.observation_system = observation_system
        self.p2_obs_system = p2_obs_system

        self.pointer: int

    def reset(self):
        '''
        Clean the history.
        '''
        self.p1_cards = [NullCard() for _ in range(self.half_total_cards)]
        self.p2_cards = [NullCard() for _ in range(self.half_total_cards)]
        self.p1_first = [False] * self.half_total_cards
        self.pointer = 0

    def add(self, p1: Card, p2: Card, p1_first: bool):
        '''
        Add cards to the history.
        '''
        self.p1_cards[self.pointer] = p1
        self.p2_cards[self.pointer] = p2
        self.p1_first[self.pointer] = p1_first
        self.pointer += 1

        self.observation_system.update_history(p1, p2, p1_first)
        
        if self.p2_obs_system != None:
            self.p2_obs_system.update_history(p1, p2, p1_first)

class Bisca:
    def __init__(self, player2: Player, observation_system: Observation = None, verbose=False) -> None:
        self.player1 = EnvPlayer()
        self.player2 = player2
        self.deck = Deck()
        
        self.scores: dict[str, int]
        self.table: Card
        self.bisca: Card

        if observation_system == None:
            self.observation_system = Observation()
        else:    
            self.observation_system = observation_system
        
        if player2.has_observation:
            self.history: History = History(self.observation_system, player2.observation_system)
        else:
            self.history: History = History(self.observation_system)

        self.verbose = verbose

    def reset(self, play_first=True, cards_order:list[int]=None) -> np.ndarray:
        self.deck.reset(cards_order)
        self.observation_system.reset()
        if self.player2.has_observation:
            self.player2.observation_system.reset()

        self.scores = {"p1": 0, "p2": 0}
        self.bisca = self.deck.last_card()
        self.player1.hand = Hand(self.deck.take_cards(3))
        self.player2.hand = Hand(self.deck.take_cards(3))

        self.history.reset()
        
        self.table = NullCard()
        if not play_first:
            self.table = self.player2.chose_card(self.table, self.bisca, self.history)

        observation = self.observation_system.get_observations(self.player1.hand, self.table, self.bisca, self.history)
        return observation

    def step(self, action: int):
        player1_card = self.player1.chose_card(action) 
        
        if self.table.is_null: # Player 1 played first
            self.table = player1_card
            player2_card = self.player2.chose_card(self.table, self.bisca, self.history)

            self.history.add(player1_card, player2_card, p1_first=True)
            played_cards = {"first": player1_card, "second": player2_card} 
            
            has_won = self.winner(player1_card, player2_card, self.bisca) == 1
        else: # Player 2 played first
            player2_card = self.table
            
            self.history.add(player1_card, player2_card, p1_first=False)
            played_cards = {"first": player2_card, "second": player1_card} 
            
            has_won = self.winner(player2_card, player1_card, self.bisca) == 0

        if self.verbose:
            print("P2:", player2_card)

        round_winner = "p1"
        done = self.player1.has_empty_hand()
        round_points = player1_card.points + player2_card.points 
        if not has_won:
            round_winner = "p2"
            if not done:
                self.player2.add_card(self.deck.take_cards())
                self.player1.add_card(self.deck.take_cards())

                self.table = self.player2.chose_card(self.table, self.bisca, self.history)
            else:
                self.table = NullCard()

            self.scores["p2"] += round_points 
        else:
            if not done:
                self.player1.add_card(self.deck.take_cards())
                self.player2.add_card(self.deck.take_cards())
                self.table = NullCard()
            
            self.scores["p1"] += round_points 
        
        obs = self.observation_system.get_observations(self.player1.hand, self.table, self.bisca, self.history)
        reward = rewards.CardPoints.get_reward(player1_card, player2_card, has_won, done, self.scores, self.bisca)

        info = {
            "played_cards": played_cards,
            "scores": self.scores,
            "winner": round_winner,
            "num_hand_cards": len(self.player1.hand.available_cards)
        }

        return obs, reward, done, info
    
    @staticmethod
    def winner(card1: Card, card2: Card, bisca: Card) -> int:
        '''
        Check which card won, given the bisca:

        Return:
        -------
            1: If card1 has won.
            0: If card2 has won. 
        '''
        is_c1_bisca = card1.suit == bisca.suit
        is_c2_bisca = card2.suit == bisca.suit

        if not is_c2_bisca:
            if is_c1_bisca:
                return 1
            
            if card1.suit != card2.suit:
                return 1

            return int(card1.is_bigger(card2))
        else:
            if is_c1_bisca:
                return int(card1.is_bigger(card2))
            
            return 0
        
    def print_current_state(self):
        print("Mão: ", end="")
        for c in self.player1.hand:
            print(f"{c} | ", end="")
        print()

        print("Bisca:", self.bisca)
        print("Mesa:", self.table)
    
    def print_player2_hand(self):
        print("PLayer 2:", self.player2.hand)

if __name__ == "__main__":
    # cards: list[Card] = []
    # num_cards = 5
    # for n, s in zip(np.random.randint(1, 13, size=num_cards), np.random.randint(0, len(Suit.all), size=num_cards)):
    #     cards.append(Card(n, s))
    
    # print("Mão")
    # for c in cards[:3]:
    #     print(c, c.internal_str())
    # print("\nBisca")
    # print(cards[3], cards[3].internal_str())
    # print("\nMesa")
    # print(cards[4], cards[4].internal_str())

    # obs = Observation.get_observations(hand=cards[:3], bisca=cards[3], table=cards[4], history=[])
    
    # print(obs.reshape(5, 5))

    import bisca_models.models as models
    import bisca_env.players as players

    model = models.Random()
    env = Bisca(players.Random(), verbose=True)
    obs = env.reset()

    done = False
    while not done:
        action = model.get_action(obs)
        
        print("Mão:")
        for c in env.player1.hand:
            print(c)
        print()

        print("Bisca:", env.bisca)
        print("Mesa:", env.table)
        print("Ação:", env.player1.hand[action])
        input()

        new_obs, reward, done = env.step(action)
        obs = new_obs

    unique_cards_history = True
    for id, c in enumerate(env.history.cards):
        if c in env.history.cards[:id] or c in env.history.cards[id+1:]:
            unique_cards_history = False
            break

    only_non_null = not (NullCard() in env.history.cards)
