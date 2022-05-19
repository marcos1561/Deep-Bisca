import tensorflow as tf
import random

from custom_env import Bisca
from custom_env import np


def test_choose_play(states, models, model_names=None):
    '''
      Test how the models responds in given states

      Parameters:
          states: list
              list of states in the human form.

          models: list
              list of models.

          models_names: list
              list of strings containing models names. 
    '''
    m_states = Bisca.human_to_machine(states)

    if model_names == None:
        model_names = []
        for i in range(len(models)):
            model_names.append(f"modelo_{i+1}")

    for s in m_states:
        hand = s[Bisca.HAND_IDS]

        actions = []
        predictions = []
        for model in models:
            # find number of cards played in the current state
            num_cards_played = 0
            for card in hand:
                if card[-1] == -1:
                    num_cards_played += 1
                else:
                    break

            if model == "Random":
                possible_actions = Bisca.ACTION_SPACE[num_cards_played:]
                action = np.random.choice(possible_actions)
            else:
                state_reshaped = Bisca.reshape_state(s)
                state_reshaped = state_reshaped.reshape([1, state_reshaped.size])
                predicted = model.predict(state_reshaped, verbose=0).flatten()
                max_p = predicted[num_cards_played:].max()
                action = np.where(predicted == max_p)[0][0]
               
                actions.append(action)
                predictions.append(predicted)

        # Print current state and models played cards.
        Bisca.print_state(s)
        print()

        max_name_length = max([len(m) for m in model_names])

        linha1 = "Modelo".center(max_name_length+1) + "| Carta Jogada | " +  "Predição".center(28)
        sep_line = "-"*(max_name_length+1) + "|" + "-"*14 + "|" + "-"*30
        print(linha1, sep_line, sep="\n")
        for m_name, action, pred in zip(model_names, actions, predictions):
            print(m_name.ljust(max_name_length+1), "| ", f"{Bisca.printable_card(hand[action]):<13}", "| ", f"{pred}", sep="")
            print(sep_line)
        print("\n", "#"*20, "\n", sep="")


def test_against_model(total_eps, model1="Random", model2="Random"):
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
    env = Bisca()

    wins = 0
    num_ep = 0

    # Random bool list for randomize who plays first
    first_move_list = np.random.random(total_eps) > 0.5
    # Frequency to show progress
    freq_prog = int(total_eps/10)
    while num_ep < total_eps:
        num_ep += 1
        if num_ep % freq_prog == 0:
            print(f"Progresso: {num_ep/total_eps*100:.0f} % | pc_win = {wins/num_ep*100:.2f}")

        observation = env.reset(first_move=first_move_list[num_ep-1], model=model2)
        done = False
        while not done:
            if model1 == "Random":
                action = np.random.choice(env.ACTION_SPACE[env.num_cards_played[0]:])
            else:
                action = env.get_action(model1, player=0)

            new_state, reward, done, info = env.step(action, model=model2)

            if done:
                if env.count_round_win[0] > env.count_round_win[1]:
                    wins += 1

    print(f"Porcentam de vitória: {wins/total_eps*100:.2f}")


def play_against(first_move, model, names=["jogador_1", "jogador_2"], show_model_hand=False, verbose=0):
    '''
        Play against model.

        Parameters:
            first_move: bool
                If true, tou will play first.

            model: tf model or "Random"

            names: list of str
                Names for you (first element) and the model (second element)

            verbose: int
                If 0 no verbose, if 1 verbose.
    '''
    env = Bisca()

    wins = 0

    num_ep = 0
    while True:
        num_ep += 1
        round_count = 0
        first_move = random.randint(0, 1)
        state = env.reset(first_move=first_move, model=model)

        print(f"### PARTIDA {num_ep} ###")

        done = False
        while not done:
            round_count += 1
            print(f"Rodada {round_count}", "-"*20, sep="\n")
            env.print_state(state)
            if show_model_hand:
                env.print_state(env.current_state[1], only_cards=True)

            valid_action = False
            while not valid_action:
                action = int(input("Acao: ")) - 1
                possible_actions = list(Bisca.ACTION_SPACE[env.num_cards_played[0]:])
                possible_actions.append(-2)
                valid_action = action in possible_actions
            print()

            if action == -2:
                return

            new_state, reward, done, info = env.step(action, model, model_name=names, verbose=1)

            if verbose:
                print(f"\nreward: {reward} | done:{done}\n")

            state = new_state

            if done:
                if env.count_round_win[0] > env.count_round_win[1]:
                    wins += 1

                print(f"Placar: {names[0]} {env.count_round_win[0]} X {names[1]} {env.count_round_win[1]}")
                print(f"Porcentagem de vitórias de {names[0]}: {wins/num_ep*100:.2f}\n")


if __name__== "__main__":
    model1_name = "bianca_v5"
    model2_name = "bianca_v2"

    model1 = tf.keras.models.load_model("saved_model/" + model1_name)
    model2 = tf.keras.models.load_model("saved_model/" + model2_name)

    models = [model1, model2]
    model_names = [model1_name, model2_name]

    states = ["9-p 5-e 1-e 1-p 2-c VAZIO- VAZIO- VAZIO- VAZIO- -1 -1", 
            "9-p 5-e 1-e 3-p 2-c VAZIO- VAZIO- VAZIO- VAZIO- -1 -1",
            "9-p 5-e 1-e 12-p 2-c VAZIO- VAZIO- VAZIO- VAZIO- -1 -1",
            "9-p 5-e 1-e 8-p 2-c VAZIO- VAZIO- VAZIO- VAZIO- -1 -1"]
    # states = ["10-e 5-e 9-e 1-p 2-c VAZIO- VAZIO- VAZIO- VAZIO- -1 -1",
    #         "3-o 4-o 2-o 12-o 1-o VAZIO- VAZIO- VAZIO- VAZIO- -1 -1",
    #         "3-o 4-p 2-o 12-o 1-o VAZIO- VAZIO- VAZIO- VAZIO- -1 -1"]


    # test_choose_play(states, models, model_names)
    # print(Bisca.human_to_machine([states[-1]]))
    play_against(True, model1, ["marcos", "bianca"], show_model_hand=True)
    # test_against_model(2000, model1, model2)
