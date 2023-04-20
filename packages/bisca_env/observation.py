import numpy as np

from bisca_env.bisca_components import Suit, Card, NullCard, Deck
from bisca_env.player_hand import Hand

class Observation:
    '''
    Controls how to the the data that will be used in the NT is structured.
    '''
    name = "without_history"

    suit_to_input_suit = {
        Suit.espadas: [1, 0, 0, 0],
        Suit.ouro: [0, 1, 0, 0],
        Suit.copas: [0, 0, 1, 0],
        Suit.paus: [0, 0, 0, 1],
        Suit.null: [0, 0, 0, 0],
    }
    non_zero_entry_id_to_suit = {0: Suit.espadas, 1: Suit.ouro, 2: Suit.copas, 3: Suit.paus}

    input_suit_shape = 4
    input_number_shape = 1
    input_card_shape = input_suit_shape + input_number_shape 

    OBSERVATION_N = (input_card_shape*5,)

    ACTION_SPACE = np.array([0, 1, 2])
    ACTION_N = ACTION_SPACE.size

    def __init__(self):
        self.base_observation = np.zeros(Observation.input_card_shape * 5)

        self.hand_init_id = []
        self.hand_final_id = []
        for i in range(3):
            init_id = Observation.input_card_shape*i
            final_id = init_id + Observation.input_card_shape
            self.hand_init_id.append(init_id)
            self.hand_final_id.append(final_id)
        
        init_id = Observation.input_card_shape*3
        final_id = init_id + Observation.input_card_shape
        self.bisca_ids = [init_id, final_id]
        
        init_id = Observation.input_card_shape*4
        final_id = init_id + Observation.input_card_shape
        self.table_ids = [init_id, final_id]

        self.all_input_cards = {NullCard().name: Observation.input_card(NullCard())}
        for c in Deck.cards:
            self.all_input_cards[c.name] = Observation.input_card(c)

    @staticmethod
    def input_suit(suit: int):
        '''
        Machine representation of a suit.

        Each suit is represented as a unit 4-d vector with only one non zero entry.
        '''
        return Observation.suit_to_input_suit[suit]
    
    @staticmethod
    def input_number(number: int) -> int:
        '''
        Machine representation of the number of a card.

        The machine representation is between 0 and 1, where the more powerful
        a card is, the closest to 1 this number is.
        '''
        if number == -1:
            return 0

        return Card.card_number_to_power_number[number]/11

    @staticmethod
    def input_card(card: Card):
        '''
        Machine representation of a card.
        '''
        return Observation.input_suit(card.suit) + [Observation.input_number(card.number)]

    def get_observations(self, hand: list[Card], table: Card, bisca: Card, history: list[Card]=[]) -> np.ndarray:
        '''
        Given the current state of the game, returns the machine representation of this state. 
        '''
        Observation.sort_hand(hand)

        for i in range(3):
            # init_id = Observation.input_card_shape*i
            # final_id = init_id + Observation.input_card_shape
            # observation[init_id:final_id] = Observation.input_card(hand[i])
            self.base_observation[self.hand_init_id[i]:self.hand_final_id[i]] = self.all_input_cards[hand[i].name]


        # init_id = Observation.input_card_shape*3
        # final_id = init_id + Observation.input_card_shape
        # observation[init_id:final_id] = Observation.input_card(bisca)
        self.base_observation[self.bisca_ids[0]:self.bisca_ids[1]] = self.all_input_cards[bisca.name]
        
        # init_id = Observation.input_card_shape*4
        # final_id = init_id + Observation.input_card_shape
        # observation[init_id:final_id] = Observation.input_card(table)
        self.base_observation[self.table_ids[0]:self.table_ids[1]] = self.all_input_cards[table.name]

        return self.base_observation.reshape(1, -1).copy()

    @staticmethod
    def sort_hand(hand: Hand):
        '''
        Sort the hand to avoid different hands with the same cards.
        '''
        has_swapped = True 
        while has_swapped:
            has_swapped = False
            for i in range(2):
                c1, c2 = hand[i], hand[i+1]
                to_swap = False 

                if c2.is_bigger(c1):
                    to_swap = True
                elif c2.is_order_equal(c1):
                    if c2.suit > c1.suit:
                        to_swap = True

                # is_c2_bisca = c2.suit == bisca.suit
                # is_c1_bisca = c1.suit == bisca.suit
                # if is_c2_bisca:
                #     if not is_c1_bisca:
                #         to_swap = True 
                #     elif c2.is_bigger(c1):
                #         to_swap = True 
                # elif not is_c1_bisca:
                #     if c2.is_bigger(c1):
                #         to_swap = True

                if to_swap:
                    has_swapped = True
                    hand[i], hand[i+1] = c2, c1 

    @staticmethod
    def non_null_cards(observation: np.ndarray):
        '''
        Returns the IDs of the non-empty cards in the `observation`.
        '''
        observation = observation[0]
        ids = []

        for i in range(3):
            init_id = i * Observation.input_card_shape
            final_id = init_id + Observation.input_suit_shape
            if observation[init_id: final_id].sum() == 1:
                ids.append(i)
        
        return ids

    @staticmethod 
    def inverse_input_number(input_number: int) -> int:
        return Card.power_number_to_card_number[int(input_number * 11)]

    @staticmethod
    def inverse_input_suit(input_suit: np.ndarray) -> int:
        for id, i in enumerate(input_suit):
            if i == 1:
                return Observation.non_zero_entry_id_to_suit[id]
        return Suit.null

    @staticmethod
    def human_card(input_card: np.ndarray) -> Card:
        input_suit = input_card[:Observation.input_suit_shape]
        input_number = input_card[Observation.input_suit_shape]

        if input_card[:Observation.input_suit_shape].sum() == 0:
            return NullCard()

        suit = Observation.inverse_input_suit(input_suit)
        number = Observation.inverse_input_number(input_number)

        return Card(number, suit)

    @staticmethod
    def get_human_cards(observation: np.ndarray):
        '''
        Return:
        -------
            hand:

            bisca:

            table:
        '''
        observation = observation[0]
        cards = []
        for i in range(5):
            init_id = i*Observation.input_card_shape
            final_id = init_id + Observation.input_card_shape
            input_card = observation[init_id: final_id]
            cards.append(Observation.human_card(input_card))
        
        hand = cards[:3]
        bisca = cards[3]
        table = cards[4]

        return hand, bisca, table

    def observation_from_str(self, obs_str: dict[str: str]) -> np.ndarray:
        cards_str = obs_str["mao"] + "," + obs_str["mesa"] + "," + obs_str["bisca"]
        cards = []
        for str_card in cards_str.split(","):
            cards.append(Card.card_from_str(str_card))

        hand = cards[:3]
        table = cards[3]
        bisca = cards[4]

        obs = self.get_observations(hand, table, bisca)
        return obs, [str(c) for c in hand]

    def update_history(self, p1_card: Card, p2_card: Card, p1_first: bool):
        pass

    def reset(self):
        pass

class History(Observation):
    name = "long_history"

    num_history_cards = 5
    
    init_history_id = Observation.input_card_shape * 5
    cards_history_ids = {}
    id = 0
    for suit in Suit.all:
        for number in [10, 11, 12, 3, 1]:
            cards_history_ids[Card(number, suit).name] = id
            id += 1

    OBSERVATION_N = (Observation.input_card_shape*5 + Suit.num * num_history_cards,)

    def __init__(self) -> None:
        super().__init__()    
        self.base_observation = np.zeros(Observation.input_card_shape * 5 + History.num_history_cards * Suit.num)
        self.cards_ids = [0, Observation.input_card_shape * 5]

    def reset(self):
        self.base_observation = np.zeros(Observation.input_card_shape * 5 + History.num_history_cards * Suit.num)

    def update_history(self, p1_card: Card, p2_card: Card, p1_first: bool=None):
        if p1_card.points > 0:
            self.base_observation[self.init_history_id + self.cards_history_ids[p1_card.name]] = 1
        if p2_card.points > 0:
            self.base_observation[self.init_history_id + self.cards_history_ids[p2_card.name]] = 1

    # def get_observations(self, hand: list[Card], table: Card, bisca: Card, history: list[Card]=[]) -> np.ndarray:
    #     '''
    #     Given the current state of the game, returns the machine representation of this state. 
    #     '''
    #     self.base_observation[self.cards_ids[0]:self.cards_ids[1]] = super().get_observations(
    #         hand, table, bisca
    #     )[0]

    #     return self.base_observation.reshape(1, -1).copy()

observation_collection = {
    "without_history" : Observation,
    "long_history": History,
}

if __name__ == "__main__":
    states = [
        {
            "mao": "10-e, 3-p, 5-c",
            "mesa": "7-p", 
            "bisca": "6-e"
        },
        {
            "mao": "10-e, 3-p, 5-c",
            "mesa": "v", 
            "bisca": "6-e"
        },
    ]

    Observation.observation_from_str(states[0])

    # cards = [
    #     Card(4, Suit.espadas), 
    #     NullCard(), 
    #     Card(6, Suit.espadas), 
    # ]

    # hand = Hand(cards)

    # print(hand)
    # Observation.sort_hand(hand)
    # print(hand)
