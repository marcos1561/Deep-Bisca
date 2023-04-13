import random
import numpy as np
from abc import ABC, abstractmethod
from bisca_components import Card, NullCard
from observation import Observation
from player_hand import Hand

class OutPLayer(ABC):
    def __init__(self, observation_system: Observation = None) -> None:
        self.observation_system = observation_system

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> int:
        pass

class RandomOut(OutPLayer):
    def get_action(self, observation: np.ndarray) -> int:
        cards = self.observation_system.non_null_cards(observation)
        return random.choice(cards)

class Player(ABC):
    def __init__(self, hand_cards: list[Card] = [NullCard(), NullCard(), NullCard()]) -> None:
        '''
        Initialize the player hand
        '''
        self.hand = Hand(hand_cards)

        self.has_observation = False

    @abstractmethod
    def chose_card(self, table: Card=None, bisca: Card=None, history: list[Card]=None) -> Card:
        '''
        Choses a card to be played given the current state of the game.
        '''
        pass

    def add_card(self, card: Card):
        '''
        Add a card to the hand cards.
        '''
        self.hand.add_card(card)

    def print_hand(self):
        for card in self.hand:
            print(card, " | ", end="")
        print()

    def has_empty_hand(self):
        '''
        Returns True is all hand cards are empty. False otherwise.
        '''
        for c in self.hand:
            if not c.is_null:
                return False

        return True

class EnvPlayer(Player):
    '''
    Player used as player 1 in the bisca environment.

    Its `chosen_card` only takes one argument, which is the card_id to chose. Normally, the
    card_id is the action. 
    '''
    def chose_card(self, card_id: int) -> Card:
        return self.hand.take_card(card_id)

class Random(Player):
    '''
    Randomly chose a card from the hand.
    '''
    def chose_card(self, table: Card = None, bisca: Card = None, history: list[Card] = None) -> Card:
        chosen_id = random.choice(self.hand.available_cards)
        return self.hand.take_card(chosen_id)

class FirstCard(Player):
    def __init__(self, observation_system: Observation) -> None:
        super().__init__()
        self.obs_system = observation_system

    def get_action(self, observation: np.array):
        return 0
    
    def chose_card(self, table: Card = None, bisca: Card = None, history: list[Card] = None) -> Card:
        self.obs_system.sort_hand(self.hand)
        return self.hand.take_card(0)

class MarcosStrat(Player):
    null_card = NullCard()

    def __init__(self, observation_system: Observation = None) -> None:
        super().__init__()
        self.observation_system = observation_system

    def get_action(self, observation: np.ndarray):
        hand, bisca, table = self.observation_system.get_human_cards(observation)

        self.hand = Hand(hand)

        if table == NullCard():
            card_id_chosen = self.empty_table(bisca)
        else:
            card_id_chosen = self.non_empty_table(table, bisca)

        while self.hand[card_id_chosen] == self.null_card:
            card_id_chosen -= 1

        return card_id_chosen

    def chose_card(self, table: Card=None, bisca: Card=None, history: list[Card]=None) -> Card:
        '''
        Choses a card to be played given the current state of the game.
        '''
        self.sort_hand()

        if table == NullCard():
            card_id_chosen = self.empty_table(bisca)
        else:
            card_id_chosen = self.non_empty_table(table, bisca)

        while self.hand[card_id_chosen] == self.null_card:
            card_id_chosen += 1

        return self.hand.take_card(card_id_chosen)

    def empty_table(self, bisca: Card):
        for id, c in enumerate(self.hand):
            if c.suit != bisca.suit:
                return id
        
        return 0

    def non_empty_table(self, table: Card, bisca: Card):
        if table.suit == bisca.suit:
            if table.points == 10:
                for id, c in enumerate(self.hand):
                    if c.points == 11 and c.suit == table.suit:
                        return id
            
            return self.get_small_card(bisca)
        else:
            if table.points > 0:
                bigger_win_card = self.bigger_win_card(table)
                if bigger_win_card != None:
                    return bigger_win_card

                if table.points > 9:
                    lowest_bisca = None
                    for id, c in enumerate(self.hand):
                        if c.suit == bisca.suit:
                            lowest_bisca = id
                            break

                    if lowest_bisca != None:
                        return lowest_bisca
            
            return self.get_small_card(bisca)

    def get_small_card(self, bisca: Card):
        small_non_bisca_card = None
        small_card = None
        for id, c in enumerate(self.hand):
            if c.points < 5: 
                if small_card == None:
                    small_card = id

                if c.suit != bisca.suit:
                    small_non_bisca_card = id
                    break

        if small_non_bisca_card != None:
            return small_non_bisca_card
        if small_card != None:
            return small_card

        return 0

    def bigger_win_card(self, card: Card):
        '''
        The hand must be sorted.
        '''
        bigger_win_card = None
        for id, c in enumerate(self.hand):
            if c.suit == card.suit and c.is_bigger(card):
                bigger_win_card = id 
        return bigger_win_card

    def sort_hand(self):
        '''
        Sort the hand such that, the higher the index of the card, the moere powerful it is.
        '''
        has_swapped = True 
        while has_swapped:
            has_swapped = False
            for i in range(2):
                c1, c2 = self.hand[i], self.hand[i+1]
                to_swap = False 

                if c1.is_bigger(c2):
                    to_swap = True

                if to_swap:
                    has_swapped = True
                    self.hand[i], self.hand[i+1] = c2, c1 

class NeuralNetwork(Player):
    def __init__(self, model, observation_system: Observation) -> None:
        super().__init__()
        self.model = model 
        self.observation_system = observation_system

        self.has_observation = True

    def get_action(self, observation: np.ndarray):
        # predicted = self.model.predict(observation, verbose=0).flatten()
        predicted = self.model(observation).numpy()[0]
        

        num_available_cards = len(self.observation_system.non_null_cards(observation))
        action = np.argmax(predicted[:num_available_cards])
        return action
    
    def random_action(self, observation: np.ndarray):
        non_null_cards = self.observation_system.non_null_cards(observation)
        return random.choice(non_null_cards) 

    def chose_card(self, table: Card = None, bisca: Card = None, history: list[Card] = None) -> Card:
        obs = self.observation_system.get_observations(self.hand, table, bisca, history)
        action = self.get_action(obs)
        return self.hand.take_card(action)


if __name__ == "__main__":
    # cards = [Card(4, 0), Card(11, 0), Card(6, 2)]
    # hand = Hand(cards)
    
    # for c in hand:
    #     print(c)
    # print(len(hand))
    # print(random.choice(hand))

    all_input_cards = [Card(1, 0), Card(11, 0), Card(6, 2)]
    hand = Hand(all_input_cards)

    p = Random(hand)
    p.print_hand()
    print(p.chose_card())
    print(p.chose_card())
    p.print_hand()
    p.add_card(Card(4, 3))
    p.print_hand()

