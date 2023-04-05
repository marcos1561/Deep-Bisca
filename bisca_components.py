import random

class Suit:
    '''
    Names for the internal representations of suits in Bisca.
    '''
    copas= 0
    cups= 0

    espadas= 1
    swords= 1

    paus= 2
    clubs= 2

    ouro= 3
    coins= 3

    null= -1

    all = [0, 1, 2, 3]
    num = 4

    # String suit to internal suit
    str_suit_to_internal = {"c": 0, "e": 1, "p": 2, "o": 3}

    internal_to_name = {0: "copas", 1: "espadas", 2: "paus", 3: "ouro"}
    @staticmethod
    def name(internal: int) -> str:
        "Human name of the suit represented internally by `internal`."
        return Suit.internal_to_name[internal]

class Card:
    # Card number to its value in game. 
    cards_number_to_points = {1: 11, 3: 10, 12: 4, 11: 3, 10: 2} 
    
    # Card number to its power number. Ignoring suits, cards with highest power number
    # beat cards with lower power number.    
    card_number_to_power_number = {1:11,2:0,3:10,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9} 
    power_number_to_card_number = dict(zip(card_number_to_power_number.values(), card_number_to_power_number.keys()))

    def __init__(self, number: int, suit: int) -> None:
        self.number = number
        self.suit = suit
        self.name = str(self.number) + "-" + Suit.name(self.suit)

        self.points = Card.cards_number_to_points.get(number, 0)
        self.power_number = Card.card_number_to_power_number[number] 

        self.is_null = False

    def is_bigger(self, other):
        '''
        Returns true is this card is powerful than the `other`. False otherwise.
        '''
        return self.power_number > other.power_number

    def is_order_equal(self, other):
        '''
        Returns true is this card has the same power number than the `other`. False otherwise.
        '''
        return self.power_number == other.power_number

    def internal_str(self):
        '''
        String with the internal representation of this cards number and suit.
        '''
        return f"{self.power_number}-{self.suit}"

    @staticmethod
    def card_from_str(str_card: str):
        str_card = str_card.strip()

        if str_card == "v":
            return NullCard()
        
        number, suit = str_card.split("-")

        number = int(number)
        suit = Suit.str_suit_to_internal[suit]

        return Card(number, suit)

    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return self.number == other.number and self.suit == other.suit 

class NullCard(Card):
    '''
    Represents an empty card
    '''
    def __init__(self) -> None:
        self.number = -1
        self.power_number = -1
        self.suit = Suit.null
        self.is_null = True
        self.name = "Vazio"
        self.points = -1
    
class Deck:
    num_total_cards = len(Card.card_number_to_power_number.keys()) * len(Suit.all)
    
    cards: list[Card] = []
    for number in Card.card_number_to_power_number.keys():
        for suit in Suit.all:
            cards.append(Card(number, suit))
    num_cards = len(cards)

    def __init__(self) -> None:
        '''
        Bisca Deck.
        '''
        self.cards_order = list(range(len(self.cards))) 
        self.next_card_id: int
        self.reset()

    def reset(self, cards_order: list[int] = None):
        '''
        Shuffles the cards and sets the next card to be picked up at the top of the deck.
        '''
        if cards_order == None:
            random.shuffle(self.cards_order)
        else:
            self.cards_order = cards_order

        self.next_card_id = 0

    def take_cards(self, num=1) -> list[Card] | Card:
        '''
        Take `num` cards from the deck and return them.

        If there is no more cards in the deck, returns a null card.
        
        Return:
        -------
        take_out_cards:
            Cards taken from the deck. First element is the first card taken and so on.

            If there is only one card taken, only this card is returned (It is not in a list).
        '''
        take_out_cards = []
        for _ in range(num):
            if self.next_card_id+1 > self.num_cards:
                card = NullCard()
            else:
                card = self.cards[self.cards_order[self.next_card_id]]
                self.next_card_id += 1
            
            take_out_cards.append(card)
        
        if num == 1:
            return take_out_cards[0]
        else:
            return take_out_cards
    
    def last_card(self) -> Card:
        '''
        Last card of the deck.
        '''
        return self.cards[self.cards_order[-1]]

if __name__ == "__main__":
    a = [Card(4, Suit.espadas), Card(12, Suit.copas)]
    b = [Card(5, Suit.espadas), Card(12, Suit.copas)]

    print(Card(4, Suit.espadas) == Card(4, Suit.espadas))