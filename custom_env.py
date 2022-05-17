from tabnanny import verbose
import numpy as np

class Bisca():
    NULL_CARD = np.array([0, 0, 0, 0, -1])
    HAND_SORT_CONST = np.array([1, 3, 5, 7, 1])
    WINNER_TO_PLAYER_ID = {1: 0, -1: 1}

    CARD_IDX = np.linspace(0, 1, 12)
    IDX_TO_CARDS = dict(zip(CARD_IDX, [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 1]))
    IDX_TO_CARDS[-1] = "VAZIO"
    CARDS_TO_IDX = dict(zip(IDX_TO_CARDS.values(), IDX_TO_CARDS.keys()))

    SUIT_IDX = np.array([1, 2, 3, 4])
    IDX_TO_SUIT = {0: "", 1: "-C", 2: "-E", 3: "-O", 4: "-P"}
    SUIT_TO_IDX = dict(zip(["", "c", "e", "o", "p"], IDX_TO_SUIT.keys()))

    OBSERVATION_N = (9*5 + 2,)
    OBSERVATION_SIZE = 9*5 + 2
    OBSERVATION_SHAPE = (11, 5)

    ACTION_SPACE = np.array([0, 1, 2])
    ACTION_N = ACTION_SPACE.size

    HAND_IDS = [2,3,4]
    PAST_CARDS_IDS = [[5,6], [7,8]]
    FIRST_MOVE_IDS = [9, 10]
    RESHAPED_FIRST_MOVE_IDS = [FIRST_MOVE_IDS[0] * 5, FIRST_MOVE_IDS[1] * 5]

    def reset(self, first_move, model=None):
        '''
            Here, when a variable contains information about both players
            the first index (0) if for player 1 and the second (1) is
            for player 2.

            Variables:
                current_state: list
                    Current states for the players.

                    A state is a list of cards.

                    A card is represent by an np-array with 5 elements:
                        The first 4 elements represent the suit and can be any 4_D identity vector. 
                        The 5-th element is a number between 0 and 1 representing the card number.

                    State has the following structure:
                        1° idx: Card in table
                        2° idx: Bisca
                        3° - 5°: Hand

                count_roud_win: list
                    Contém a quantidade de rodadas vencidas por cada jogador em uma partida

                num_cards_played: list
                    Contém a quantidade de cartas jogadas até o momento para cada jogador 
                    na atual partida.

                A card is represent by an array with 5 elements:
                    The first 4 elements represent the suit and can be any 4_D identity vector. 
                    The 5-th element is a number between 0 and 1 representing the card number. 

                State has the following structure
                    1° idx: Table
                    2° idx: Bisca
                    3° - 5°: Hand

                Action has the following structure
        '''
        self.count_round_win = [0, 0]
        self.num_cards_played = [0, 0]

        ## Construct deck ###
        deck = np.zeros((12*4, 5))

        for i in range(4):
            id_i = 12*i
            id_f = id_i + 12
            deck[id_i:id_f, i] = 1
            deck[id_i:id_f, -1] = Bisca.CARD_IDX

        deck_idx = list(range(12*4))
        ######

        ### Setting initial state ###
        # first cards index
        init_cards_idx = np.random.choice(deck_idx, 7, replace=False)
        for i in init_cards_idx:
            deck_idx.remove(i)

        init_state_1 = np.zeros((11, 5))
        init_state_1[0, -1] = -1
        init_state_1[1:Bisca.HAND_IDS[-1]+1, :] = deck[init_cards_idx[:4]]
        init_state_1[Bisca.HAND_IDS, :] = Bisca.sort_hand(init_state_1[Bisca.HAND_IDS, :])
        for i in range(Bisca.HAND_IDS[-1]+1,Bisca.HAND_IDS[-1]+1 + 4):
            init_state_1[i] = Bisca.NULL_CARD
        init_state_1[Bisca.FIRST_MOVE_IDS, 0] = -1

        init_state_2 = np.zeros((11, 5))
        init_state_2[0, -1] = -1
        init_state_2[1, :] = deck[init_cards_idx[0]]
        init_state_2[Bisca.HAND_IDS, :] = deck[init_cards_idx[4:]]
        init_state_2[Bisca.HAND_IDS, :] = Bisca.sort_hand(init_state_2[Bisca.HAND_IDS, :])
        for i in range(Bisca.HAND_IDS[-1]+1,Bisca.HAND_IDS[-1]+1 + 4):
            init_state_2[i] = Bisca.NULL_CARD
        init_state_2[Bisca.FIRST_MOVE_IDS,0] = -1

        if not first_move:
            self.current_state = [init_state_2, init_state_1]

            action2 = self.get_action(model, 1)
            self.make_player_move(action2, player=1, first_move=True)
        else:
            self.current_state = [init_state_1, init_state_2]
        ######

        return self.current_state[0].copy()

    def make_player_move(self, action, player, first_move):
        '''
            Make a player move specified by the action, set all the states 
            and variables according to the move and return the played card.
        '''
        played_card = self.current_state[player][Bisca.HAND_IDS[0] + action].copy()
        self.current_state[player][Bisca.HAND_IDS[0] + action] = Bisca.NULL_CARD
        self.current_state[player][Bisca.HAND_IDS, :] = Bisca.sort_hand(self.current_state[player][Bisca.HAND_IDS, :])
        self.num_cards_played[player] += 1

        if first_move:
            self.current_state[not player][0] = played_card

        return played_card

    def get_action(self, model, player):
        '''
            Get a model action for the player with index "player"

            Parameters:
                model: tf model or "Random"

                player: int
                    player idx: 0 = player 1, 1 = player 2

            Return:
                action: int
                    action chosen
        '''
        if model == "Random":
            possible_actions = self.ACTION_SPACE[self.num_cards_played[player]:]
            action = np.random.choice(possible_actions)
        else:
            state_reshaped = Bisca.reshape_state(self.current_state[player])
            state_reshaped = state_reshaped.reshape([1, state_reshaped.size])
            predicted = model.predict(state_reshaped, verbose=0).flatten()

            max_p = predicted[self.num_cards_played[player]:].max()
            action = np.where(predicted == max_p)[0][0]

        return action

    @staticmethod
    def reshape_state(state):
        state_reshaped_temp = state.reshape(Bisca.OBSERVATION_SHAPE[0] * Bisca.OBSERVATION_SHAPE[1])
        state_reshaped = np.zeros(Bisca.OBSERVATION_SIZE)

        first_move_id_reshaped = 5*Bisca.FIRST_MOVE_IDS[0]
        state_reshaped[:first_move_id_reshaped] = state_reshaped_temp[:first_move_id_reshaped]
        state_reshaped[-2:] = state_reshaped_temp[Bisca.RESHAPED_FIRST_MOVE_IDS]
        
        return state_reshaped 


    @staticmethod
    def sort_hand(hand):
        '''
          Sort and return the hand given

          Parameters:
              hand: np-array
                  An array whose elements are cards in the machine form.
        '''
        hand_values = hand.dot(Bisca.HAND_SORT_CONST)
        hand_values_to_idx = dict(zip(hand_values, [0, 1, 2]))

        hand_values.sort()

        sorted_hand = np.zeros_like(hand)
        for idx, h_v in enumerate(hand_values):
            card_idx = hand_values_to_idx[h_v]
            sorted_hand[idx] = hand[card_idx]

        return sorted_hand


    def step(self, action, model="Random", model_name=["player1", "player2"], verbose=0):
        '''
            Make a step in the environment

            Parameters:
                action: int
                    The player1 action to execute.

                model: tf model or "Random"
                    model to play as player2.

                model_name: list of str
                    Names for the models being player 1 and 2, respectively.

                verbose: int
                    0 = no verbose: 1 = verbose
        '''
        ### Play the cards ###
        first_move = not(self.current_state[0][0][-1] + 1)
        played_card_1 = self.make_player_move(action, player=0, first_move=first_move)

        if first_move:
            action2 = self.get_action(model, 1)
            played_card_2 = self.make_player_move(
                action2, player=1, first_move=False)

            played_cards = [played_card_1, played_card_2]
        else:
            played_card_2 = self.current_state[0][0]
            played_cards = [played_card_2, played_card_1]
        ######

        ### Update memory state ###
        if self.num_cards_played[0] < 3:
            self.current_state[0][Bisca.PAST_CARDS_IDS[self.num_cards_played[0]-1],:] = np.array([played_card_1, played_card_2])
            self.current_state[0][Bisca.FIRST_MOVE_IDS[self.num_cards_played[0]-1]][0] = first_move

            self.current_state[1][Bisca.PAST_CARDS_IDS[self.num_cards_played[1]-1],:] = np.array([played_card_2, played_card_1])
            self.current_state[1][Bisca.FIRST_MOVE_IDS[self.num_cards_played[1]-1]][0] = not first_move
        ######

        ### Check who wins ###
        bisca_suit = self.current_state[0][1][:4]
        suit_1 = played_cards[0][:4]
        suit_2 = played_cards[1][:4]
        num1 = played_cards[0][-1]
        num2 = played_cards[1][-1]

        if suit_1.dot(suit_2) == 1:
            if num1 > num2:
                winner = 1
            else:
                winner = -1
        else:
            if suit_2.dot(bisca_suit) == 1:
                winner = -1
            else:
                winner = 1

        if not first_move:
            winner *= -1
        ######

        if verbose:
            print(f"Primeiro a jogar: {first_move}\n")
            self.print_state(self.current_state[0])
            print()
            self.print_state(self.current_state[1])
            print()

            print(f"Cartas jogadas: {self.printable_card(played_card_1)} | {self.printable_card(played_card_2)}")

            model_name_id = 1
            if winner == 1:
                model_name_id = 0

            print(f"Vencedor: {model_name[model_name_id]}\n")

        # Rewards values
        win_round_r = 5  # round win reward
        win_ep_r = 20  # game win reward

        ### Reward the model according to the results ###
        # Check if the game has ended
        reward = 0
        sum_null_card = self.current_state[0][Bisca.HAND_IDS, -1].sum()
        done = False
        
        if sum_null_card == -3:
            done = True

            # who wins the most amount of rounds wins the game
            if self.count_round_win[0] > self.count_round_win[1]:
                reward += win_ep_r
            else:
                reward += -win_ep_r

        if winner == -1:  # Player 2 wins round
            self.count_round_win[1] += 1
            reward += -win_round_r * self.count_round_win[1]

            if not done:
                self.current_state[1][0] = Bisca.NULL_CARD
                action2 = self.get_action(model, 1)
                self.make_player_move(action2, player=1, first_move=True)
        else:  # Player 1 wins round
            self.count_round_win[0] += 1
            reward += win_round_r * self.count_round_win[0]

            if not done:
                self.current_state[0][0] = Bisca.NULL_CARD
        ######

        # Player 1 number of cards played after the action
        info = self.num_cards_played[0]

        return self.current_state[0].copy(), reward, done, info

    @staticmethod
    def printable_card(card):
        '''
            Return the human form card of the machine form card given.

            Parameter:
                card: np-array

            Return:
                human_card: str
        '''
        suit_id = card[:4].dot(Bisca.SUIT_IDX)
        suit = Bisca.IDX_TO_SUIT[suit_id]

        card_num = Bisca.IDX_TO_CARDS[card[-1]]

        human_card = f"{card_num}{suit}"
        return human_card

    @staticmethod
    def print_state(state, only_cards=False):
        '''
            print the state given in the human form.

            Parameters:
                state: list
                    list whose elements are the cards.
                only_cards: bool
                    if true only print the cards
        '''
        table = state[0]
        bisca = state[1]
        hand = state[Bisca.HAND_IDS]

        table_str = Bisca.printable_card(table)
        bisca_str = Bisca.printable_card(bisca)

        if not only_cards:
            print(f"Mesa: {table_str}")
            print(f"Bisca: {bisca_str}")
        print(f"Mão:", end=" ")
        for card in hand:
            card_str = Bisca.printable_card(card)
            print(f"{card_str} | ", end="")
        print()

    @staticmethod
    def human_to_machine(h_state_list):
        '''
            Transform a state in the human form to machine form

            A state in the human form is a string with the cards separeted by a space
            and a single card is in the form {card number}-{car suit}.

            Parameters:
                h_state_list: list
                    A list whose elements are states in the human form

            Return:
                m_state_list: list
                    A list whose elements are states in the machine form.
        '''

        m_state_list = []
        for h_state in h_state_list:
            m_state = np.zeros(Bisca.OBSERVATION_SHAPE)
            

            h_state_elements = h_state.split(" ")
            for card_id,  h_card in enumerate(h_state_elements[:-2]):
                h_num, h_suit = h_card.split("-")

                suit_idx = Bisca.SUIT_TO_IDX[h_suit]
                m_suit = [0, 0, 0, 0]
                if suit_idx > 0:
                    m_suit[suit_idx-1] = 1

                if h_num != "VAZIO":
                    h_num = int(h_num)
                m_num = Bisca.CARDS_TO_IDX[h_num]

                m_card = np.array(m_suit+[m_num])
                m_state[card_id] = m_card
            m_state[Bisca.HAND_IDS,:] = Bisca.sort_hand(m_state[Bisca.HAND_IDS,:])

            for id, is_first_move in enumerate(h_state_elements[-2:]):
                is_first_move = int(is_first_move)
                if is_first_move == 1:
                    m_state[-2 + id] = np.array([1,0,0,0,0])
                elif is_first_move == 0:
                    m_state[-2 + id] = np.array([0,0,0,0,0])
                else:
                    m_state[-2 + id] = np.array([-1,0,0,0,0])

            m_state_list.append(np.array(m_state))

        return m_state_list
