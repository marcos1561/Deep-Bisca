from turtle import color
import pygame
import random
import os
from tensorflow.keras.models import load_model

import custom_env
import cards
import labels
from managers import *

def main():
    ## Inicialize things ###
    pygame.init()
    pygame.display.set_caption("Bisca")
     
    font = pygame.font.SysFont("monospace", 15)
    ###

    ### Setup screen ###
    display_info = pygame.display.Info()
    display_shape = display_info.current_w, display_info.current_h
    game_display_shape = tuple(int(display_dim*0.8) for display_dim in display_shape)
    screen = pygame.display.set_mode(game_display_shape)
    ######

    ### Managers ###
    hand = HandManager()
    cards.HandCard.hand = hand

    screen_manager = ScreenManager(screen)
    cards.Card.screen_manager = screen_manager
    labels.Label.screen_manager = screen_manager
    #######

    ### Sprites for cards ###
    hand_sprites = [cards.HandCard("CORINGA", i) for i in range(3)]
    bisca_sprite = [cards.BiscaCard("CORINGA")]
    table_sprites = [cards.Card("CORINGA") for _ in range(2)]
    cards_sprites = hand_sprites + bisca_sprite + table_sprites
    
    hand_group = pygame.sprite.Group(*hand_sprites)
    other_group = pygame.sprite.Group(*(bisca_sprite + table_sprites))
    ######

    ### Labels ###
    # Label to inform who won each round
    labels_color = (255,255,0)
    winner_label = labels.WinnerLabel(font, (10, 10), color=labels_color)
    
    # Label to inform the percentage of wins
    upper_left_pc_vic = (10, 10 + winner_label.bianca_win_round_label.get_height() + 4)
    pc_vic_label = labels.PcWinLabel(font, upper_left_pc_vic, "Porcentagem de vitória:", labels_color)

    labels_group = pygame.sprite.Group(winner_label, pc_vic_label)
    ######

    ### Setup card's positions ###
    cs_p = 0.2
    c_ar = 0.64
    lm_p, rm_p = 1-0.25, 0.05  
    bm_p = 0.05

    position_setup = {"cs_p": cs_p, "c_ar":c_ar, "lm_p":lm_p, "rm_p":rm_p, 
                    "bm_p":bm_p, "game_display_shape":game_display_shape}

    cards.update_cards_pos(cards_sprites, position_setup)
    ######


    ### Bisca env ###
    # model to play with
    model_name = "bianca_v2"

    def show_observation(observation, screen):
        tb_c = env.printable_card(observation[0])
        table_sprites[0].update_image(tb_c, screen)
        
        bisca_c = env.printable_card(observation[1])
        bisca_sprite[0].update_image(bisca_c, screen)
        
        for id, m_card in enumerate(observation[env.HAND_IDS]):
            h_card = env.printable_card(m_card)
            hand_sprites[id].update_image(h_card, screen)

    # Initial setup
    env = custom_env.Bisca()
    model_path = os.path.join("saved_model", model_name)
    model = load_model(model_path)

    first_move = random.randint(0,1)
    observation = env.reset(first_move=first_move, model=model)
    hand.reset()

    # Show initial state
    show_observation(observation, screen)
    table_sprites[1].update_image("VAZIO", screen)
    ######

    running = True
    showing_results = False
    done = False
    clock = pygame.time.Clock()
    while running:
        events_list = pygame.event.get()
        for event in events_list:
            if event.type == pygame.QUIT:
                running = False

            if showing_results:
                if event.type == pygame.MOUSEBUTTONUP:
                    showing_results = False
                    
                    # Clear the label of who won
                    winner_label.update_winner_label(0)
                    
                    # Update the table cards for the next round
                    m_card = env.printable_card(new_state[0])
                    table_sprites[0].update_image(m_card, screen)
                    table_sprites[1].update_image("VAZIO", screen)

        if not showing_results:
            if done:
                done = False
                
                # Check who won and store that information
                pc_vic_label.games_count += 1
                if env.count_round_win[0] > env.count_round_win[1]:
                    pc_vic_label.victory_count += 1
                
                # Reset bisca environment
                first_move = random.randint(0,1)
                observation = env.reset(first_move=first_move, model=model)
                hand.reset()

                # Update win percentage label 
                pc_vic_label.update_text()

                # Display initial observation
                show_observation(observation, screen)
            
            # Check if a hand's card was clicked and notify that to "hand"
            hand_group.update(events_list, screen)

        if hand.action_chosen:
            hand.action_chosen = False
            showing_results = True

            new_state, reward, done, info = env.step(hand.action, model)
            winner = info[1]
            played_cards = (info[0][1], info[0][0])

            # Display label of who won
            winner_label.update_winner_label(winner)

            # Display played cards
            for sprite, m_card in zip(table_sprites, played_cards):
                h_card = env.printable_card(m_card)
                sprite.update_image(h_card, screen)

        screen_manager.update_screen([hand_group, other_group, labels_group])
        clock.tick(60)

if __name__=="__main__":
    main()