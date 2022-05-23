from main import pygame
import cards
import labels
import inputs
import custom_env
from managers import SceneManager, ScreenManager

import os
from tensorflow.keras.models import load_model
import random

class Game:
    def __init__(self, model_name, hand_manager, game_display_shape) -> None:
        self.env = custom_env.Bisca()
        self.font = pygame.font.SysFont("monospace", 15)

        self.observation = None
        self.new_observation = None
        self.played_cards = None
        self.winner = None
        self.done = False

        self.showing_results = False

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
        upper_left_pc_vic = (10, 10 + self.winner_label.bianca_win_round_label.get_height() + 4)
        self.pc_vic_label = labels.PcWinLabel(self.font, upper_left_pc_vic, "Porcentagem de vitória:", labels_color)

        self.labels_group = pygame.sprite.Group(self.winner_label, self.pc_vic_label)
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

        model_path = os.path.join("saved_model", model_name)
        self.model = load_model(model_path)

        self.hand_manager = hand_manager
        
        self.reset()

    def reset(self):
        ### Reset Bisca environment ###
        first_move = random.randint(0,1)
        self.observation = self.env.reset(first_move=first_move, model=self.model)
        self.hand_manager.reset()

        self.show_observation()
    
    def update(self, events_list):
        if self.hand_manager.action_chosen:
            self.execute_action()

        if not self.showing_results:
            # Check if a hand's card was clicked and notify that to "hand_manager"
            self.hand_group.update(events_list)
        else:
            for event in events_list:
                if event.type == pygame.MOUSEBUTTONUP:
                    self.next_round()

    def execute_action(self):
        self.hand_manager.action_chosen = False
        self.showing_results = True

        self.new_observation, reward, self.done, info = self.env.step(self.hand_manager.action, self.model)
        self.winner = info[1]
        self.played_cards = (info[0][1], info[0][0])

        # Update label of who won
        self.winner_label.update_winner_label(self.winner)

        # Update played cards images
        for sprite, m_card in zip(self.table_sprites, self.played_cards):
            h_card = self.env.printable_card(m_card)
            sprite.update_image(h_card)

    def next_round(self):
        self.showing_results = False
                
        # Clear the label of who won
        self.winner_label.update_winner_label(0)
        
        # Update the table cards for the next round
        if not self.done:
            m_card = self.env.printable_card(self.new_observation[0])
            self.table_sprites[0].update_image(m_card)
            self.table_sprites[1].update_image("VAZIO")
        else:
            self.round_ended()

    def round_ended(self):
        self.done = False
        
        # Check who won the game and store that information
        self.pc_vic_label.games_count += 1
        if self.env.count_round_win[0] > self.env.count_round_win[1]:
            self.pc_vic_label.victory_count += 1

        # Update win percentage label 
        self.pc_vic_label.update_text()
        
        # Reset game scene
        self.reset()
        
    def show_observation(self):
        tb_c = self.env.printable_card(self.observation[0])
        self.table_sprites[0].update_image(tb_c)
        self.table_sprites[1].update_image("VAZIO")

        bisca_c = self.env.printable_card(self.observation[1])
        self.bisca_sprite[0].update_image(bisca_c)
        
        for id, m_card in enumerate(self.observation[self.env.HAND_IDS]):
            h_card = self.env.printable_card(m_card)
            self.hand_sprites[id].update_image(h_card)

class Menu:
    scene_manager: SceneManager
    screen_manager: ScreenManager

    def __init__(self, game_scene: Game, game_display_shape) -> None:
        self.game_scene = game_scene

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

            self.game_scene.winner_label.set_you_win_label(self.player_name)
