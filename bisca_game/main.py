import pygame

import scenes
from managers import *

def main():
    ## Initialization ###
    pygame.init()
    pygame.display.set_caption("Bisca")
    ###

    ### Setup screen ###
    display_info = pygame.display.Info()
    display_shape = display_info.current_w, display_info.current_h
    game_display_shape = tuple(int(display_dim*0.8) for display_dim in display_shape)
    screen = pygame.display.set_mode(game_display_shape)
    ######

    ### Managers ###
    hand_manager = HandManager()
    scenes.cards.HandCard.hand = hand_manager

    scene_manager = SceneManager()
    scenes.Menu.scene_manager = scene_manager

    screen_manager = ScreenManager(screen)
    scenes.cards.Card.screen_manager = screen_manager
    scenes.labels.Label.screen_manager = screen_manager
    scenes.inputs.InputBox.screen_manager = screen_manager
    scenes.Menu.screen_manager = screen_manager
    #######

    ### Scenes ###
    # Game scene
    model_name = "histpry_6_checkpoint"
    game = scenes.Game(model_name, hand_manager, game_display_shape)

    # Menu scene
    menu = scenes.Menu(game, game_display_shape)
    ######

    running = True
    clock = pygame.time.Clock()
    while running:
        events_list = pygame.event.get()
        
        for event in events_list:
            if event.type == pygame.QUIT:
                running = False

        if scene_manager.current_scene == 0:
            menu.update(events_list)
        elif scene_manager.current_scene == 1:
            game.update(events_list)
        
        if scene_manager.current_scene == 0:
            screen_manager.update_screen([menu.background, menu.input_box, menu.input_label])
        elif scene_manager.current_scene == 1:
            screen_manager.update_screen([game.background, game.hand_group, game.other_group, game.labels_group])
        
        clock.tick(60)

if __name__=="__main__":
    main()