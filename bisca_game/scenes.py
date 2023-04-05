import bisca_env as bisca_env
import players as players
from  bisca_components import Card as BiscaCard
from  observation import Observation, History

import cards, labels, inputs, background
from managers import SceneManager, ScreenManager

import pygame
import os
from tensorflow.keras.models import load_model
import random

class Game:
    def __init__(self, model_name, hand_manager, game_display_shape) -> None:
        if model_name == "mark":
            player2 = players.MarcosStrat()
        elif model_name == "random":
            player2 = players.Random()
        else:
            if model_name == "checkpoint":
                model_path = os.path.join("..//checkpoint//model")
            else:
                model_path = os.path.join("..", "saved_model", model_name)
            model = load_model(model_path)
            player2 = players.NeuralNetwork(model, History())

        self.env = bisca_env.Bisca(player2)
        
        self.font = pygame.font.SysFont("monospace", 15)

        self.player_name = "-1"
        self.winner = None
        self.observation = None
        self.new_observation = None
        self.played_cards = None
        self.scores = [0, 0]
        self.done = False

        self.showing_results = False
        self.showing_end_results = False

        ### Backgroud ###
        self.background = background.Background("table_2.jpeg", game_display_shape, rotate=True)
        ######

        ### Sprites for cards ###
        self.hand_sprites = [cards.HandCard("CORINGA", i) for i in range(3)]
        self.bisca_sprite = [cards.BiscaCard("CORINGA")]
        self.table_sprites = [cards.Card("CORINGA") for _ in range(2)]
        self.cards_sprites = self.hand_sprites + self.bisca_sprite + self.table_sprites
        
        self.hand_group = pygame.sprite.Group(*self.hand_sprites)
        self.other_group = pygame.sprite.Group(*(self.bisca_sprite + self.table_sprites))
        #######

        ### Labels ###
        # Label to inform who won each round
        labels_color = (255,255,0)
        self.winner_label = labels.WinnerLabel(self.font, (10, 10), color=labels_color)
        
        # Label to inform the percentage of wins
        upper_left_pc_vic = (10, self.winner_label.rect[1] + self.winner_label.bianca_win_round_label.get_height() + 4)
        self.pc_vic_label = labels.PcWinLabel(self.font, upper_left_pc_vic, "Porcentagem de vitória:", labels_color)

        # Total games played
        upper_left_num_games = (10, self.pc_vic_label.rect[1] + self.pc_vic_label.image.get_height() + 4)
        self.num_games_label = labels.NumGames(self.font, upper_left_num_games, "Partidas jogadas: 0", labels_color) 

        self.labels_group = pygame.sprite.Group(self.winner_label, self.pc_vic_label, self.num_games_label)
        ######
        
        ### Setup card's positions ###
        cs_p = 0.2
        c_ar = 0.64
        lm_p, rm_p = 1-0.25, 0.05  
        bm_p = 0.05

        position_setup = {"cs_p": cs_p, "c_ar":c_ar, "lm_p":lm_p, "rm_p":rm_p, 
                        "bm_p":bm_p, "game_display_shape":game_display_shape}

        cards.update_cards_pos(self.cards_sprites, position_setup)
        ######

        self.hand_manager = hand_manager
        
        self.reset()

        ### For debugging ###
        self.total_cards = self.env.deck.cards
        self.all_cards_played = []
        
        self.show_model_hand = True
        ######

    def reset(self):
        ### Reset Bisca environment ###
        self.showing_end_results = False

        # Clear the label of who won
        self.winner_label.update_winner_label(0)

        first_move = random.randint(0,1)
        self.observation = self.env.reset(play_first=first_move)
        self.hand_manager.reset()

        self.show_observation(self.observation)

        ### Debugging ###
        self.all_cards_played = []
        ######
    
    def update(self, events_list):
        if self.hand_manager.action_chosen:
            self.execute_action()

        if self.showing_results:
            for event in events_list:
                if event.type == pygame.MOUSEBUTTONUP:
                    self.next_round()
        elif self.showing_end_results:
             for event in events_list:
                if event.type == pygame.MOUSEBUTTONUP:
                    self.reset()
        else:
            # Check if a hand's card was clicked and notify that to "hand_manager"
            self.hand_group.update(events_list)
        
        ### Debugging ###
        if self.show_model_hand:
            self.show_model_hand = False
            print()
            # self.env.print_current_state()
            self.env.print_player2_hand()
        ######

    def execute_action(self):
        self.hand_manager.action_chosen = False
        self.showing_results = True

        self.new_observation, reward, self.done, info = self.env.step(self.hand_manager.action)
        self.played_cards: dict[str, BiscaCard] = info["played_cards"]
        self.winner = info["winner"]
        self.scores = info["scores"]

        ### Debugging ###
        for c in self.played_cards.values():
            c = c.__str__()
            if c not in self.all_cards_played:
                self.all_cards_played.append(c)
            else:
                print("\n\ncarta repetida!\n\n")

        self.show_model_hand = True
        print("\nPontos:", self.scores)
        #######

        # Update label of who won
        self.winner_label.update_winner_label(self.winner)

        # Update played cards images
        for sprite, bisca_card in zip(self.table_sprites, self.played_cards.values()):
            card_name = cards.Card.card_name_from_bisca_card(bisca_card)
            sprite.update_image(card_name)

    def next_round(self):
        self.showing_results = False
                
        # Clear the label of who won
        self.winner_label.update_winner_label(0)
        
        # Update the table cards for the next round
        if not self.done:
            self.show_observation(self.new_observation)
        else:
            self.game_ended()

    def game_ended(self):
        self.showing_end_results = True

        # Clean cards
        for card in self.cards_sprites:
            card.update_image("VAZIO")

        # Check who won the game and store that information
        self.pc_vic_label.games_count += 1
        if self.scores["p1"] > self.scores["p2"]:
            self.pc_vic_label.victory_count += 1

        # Show who won
        end_score = f"Placar: {self.player_name} {self.scores['p1']} X {self.scores['p2']} Bianca"   
        self.winner_label.update_text(end_score)

        # Update total games label
        self.num_games_label.update_text(self.pc_vic_label.games_count)

        # Update win percentage label 
        self.pc_vic_label.update_text()

        ### Debugging ###
        erro = False

        if len(self.total_cards) != len(self.all_cards_played):
            erro = True

        for c in self.total_cards:
            c: BiscaCard
            if c.__str__() not in self.all_cards_played:
                erro = True
        print("\n\nERRO:", erro)
        ######

    def show_observation(self, observation):
        hand, bisca, table = Observation.get_human_cards(observation)

        tb_c = cards.Card.card_name_from_bisca_card(table)
        self.table_sprites[0].update_image(tb_c)
        self.table_sprites[1].update_image("VAZIO")

        bisca_c = cards.Card.card_name_from_bisca_card(bisca)
        self.bisca_sprite[0].update_image(bisca_c)
        
        for id, m_card in enumerate(hand):
            h_card = cards.Card.card_name_from_bisca_card(m_card)
            self.hand_sprites[id].update_image(h_card)

class Menu:
    scene_manager: SceneManager
    screen_manager: ScreenManager

    def __init__(self, game_scene: Game, game_display_shape) -> None:
        self.game_scene = game_scene

        ### Backgroud ###
        self.background = background.Background("table_2.jpeg", game_display_shape, rotate=True)
        ######

        ### Input text setup ###
        font_size = 100
        font = pygame.font.Font(None, font_size)
        
        width_input = game_display_shape[0]*0.7
        height_input = font.size("a")[1] + 5

        x_input = int(game_display_shape[0]/2 - width_input/2)
        y_input = int(game_display_shape[1]/2 - height_input/2)

        self.input_box = inputs.InputBox(x_input, y_input, width_input, height_input, font)
        ######

        ### Label setup ###
        x_label = x_input
        y_label = y_input - 5 - font.size("a")[1]
        self.input_label = labels.Label(font, (x_label, y_label), text="Seu nome:", color="white")
        ######

        self.player_name = None

    def update(self, events_list):
        for event in events_list:
            self.player_name = self.input_box.handle_event(event)

        if self.player_name != None:
            Menu.scene_manager.current_scene = 1
            Menu.screen_manager.has_changed = True

            self.game_scene.player_name = self.player_name
            self.game_scene.winner_label.set_you_win_label(self.player_name)
