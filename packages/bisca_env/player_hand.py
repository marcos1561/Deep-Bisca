from bisca_env.bisca_components import Card, NullCard

class Hand:
    '''
    Represents the cards that the player is holding.
    '''

    def __init__(self, cards: list[Card]) -> None:
        '''
        Hand behaves like a list, but has 2 auxiliary fields:

            `empty_spaces`: list[bool]
                The i-th entry is True is the i-th hand card is empty.
            
            `available_cards`: list[int]
                Ids of non null cards in the hand.

        This fields are automatically updated when the hand is updated through 
        the `[]` operator.   

            
        Parameters:
        -----------
            cards:
                Initial hand cards.
        '''
        if len(cards) != 3:
            raise Exception("'cards' deve ter exatamente 3 cartas.")

        self.cards = cards
        
        self.available_cards = []
        self.empty_spaces = [True] * 3
        for id, card in enumerate(cards):
            if not card.is_null:
                self.available_cards.append(id)
                self.empty_spaces[id] = False

    def take_card(self, card_id) -> Card:
        '''
        Given the `card_id` removes this card from the hand and return it.
        '''

        chosen_card = self.cards[card_id]
        self.__setitem__(card_id, NullCard())
        return chosen_card

    def add_card(self, card: Card):
        '''
        Add a card to the first empty espace available in the hand.
        '''
        for id, is_empy in enumerate(self.empty_spaces):
            if is_empy:
                self.__setitem__(id, card)
                return

        raise Exception("Não tem espaço disponível na mão")

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, key: int) -> Card:
        return self.cards[key]

    def __setitem__(self, key: int, value: Card):
        if value.is_null:
            if key in self.available_cards:
                self.available_cards.remove(key)
            self.empty_spaces[key] = True
        else:
            if key not in self.available_cards:
                self.available_cards.append(key)
            self.empty_spaces[key] = False

        self.cards[key] = value

    def __eq__(self, other: object) -> bool:
        sorted(self.available_cards)
        sorted(other.available_cards)
        return self.cards == other.cards and self.available_cards == other.available_cards and self.empty_spaces == other.empty_spaces

    def __str__(self) -> str:
        return str([str(c) for c in self.cards])
