import custom_env
import pygame
import os
from tensorflow.keras.models import load_model
import random

class Hand():
    def __init__(self) -> None:
        pass

    def reset(self):
        self.action = 0
        self.action_chosen = False
        self.hand_idx_to_action = {0:0 ,1:1, 2:2}    

class Card(pygame.sprite.Sprite):
    def __init__(self, card_name, screen):
        super().__init__()
  
        self.screen = screen
        self.name = card_name
        
        self.image = Card.load_card_image(card_name)
        self.rect = self.image.get_rect()

    @staticmethod
    def load_card_image(card_name):
        card_path = os.path.join("cards_images", card_name + ".jpg")
        card_image = pygame.image.load(card_path).convert()
        return card_image

    def choose_image(self, card_name):
        self.name = card_name
        if card_name == "VAZIO":
            self.image.fill(pygame.Color(0,0,0,0))
        else:
            self.image = Card.load_card_image(card_name)
            self.image = pygame.transform.scale(self.image, (self.rect.width, self.rect.height))

        self.screen.blit(self.image, self.rect)
        pygame.display.flip()

class BiscaCard(Card):
    def choose_image(self, card_name):
        if card_name == "VAZIO":
            self.image.fill(pygame.Color(0,0,0,0))
        else:
            self.image = Card.load_card_image(card_name)
            self.image = pygame.transform.rotate(self.image, 90)
            self.image = pygame.transform.scale(self.image, (self.rect.width, self.rect.height))

        self.screen.blit(self.image, self.rect)
        pygame.display.flip()

class HandCard(Card):
    hand = Hand()

    def __init__(self, card_name, screen, card_idx):
        super().__init__(card_name, screen)

        self.card_idx = card_idx
        self.was_clicked = False


    def update(self, events_list) -> None:
        if self.name == "VAZIO":
            return

        selected = self.was_selected(events_list)
        if selected:
            self.choose_image("VAZIO")
            
            HandCard.hand.action = HandCard.hand.hand_idx_to_action[self.card_idx]
            HandCard.hand.action_chosen = True
            HandCard.hand.played_card = self.name

            for i in range(self.card_idx):
                HandCard.hand.hand_idx_to_action[i] += 1


    def was_selected(self, events_list):
        result = False
        for event in events_list:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.rect.collidepoint(event.pos):
                    self.was_clicked = True

            if event.type == pygame.MOUSEBUTTONUP:
                is_colliding = self.rect.collidepoint(event.pos)

                if is_colliding:
                    if self.was_clicked:
                        self.was_clicked = False
                        # print(f"Collided with {self.card_idx}")
                        result = True
                elif self.was_clicked:
                    self.was_clicked = False

        return result

    
def update_cards_pos(cards_sprites, position_setup):
    cs_p = position_setup["cs_p"]
    c_ar = position_setup["c_ar"]
    lm_p, rm_p = position_setup["lm_p"], position_setup["rm_p"]  
    bm_p = position_setup["bm_p"]

    d_w, d_h = position_setup["game_display_shape"]
    
    w_eff = d_w*(lm_p - rm_p)
    h_eff = (2 + cs_p*c_ar) * w_eff / (3*(1+cs_p)*c_ar + 1)

    c_h = w_eff / (3*(1+cs_p)*c_ar + 1)
    c_w = c_ar * c_h

    for c_sprite in cards_sprites:
        c_sprite.image = pygame.transform.scale(c_sprite.image, (c_w, c_h))
        c_sprite.rect = c_sprite.image.get_rect()

    pos_y = d_h * (1 - bm_p) - c_h
    for i in range(3):
        pos_x = d_w*(1-rm_p) - c_ar * c_h - i * c_ar * c_h * (1 + cs_p)

        c_sprite = cards_sprites[i]
        c_sprite.rect.x = pos_x
        c_sprite.rect.y = pos_y


    bisca_card = cards_sprites[3]
    bisca_card.image = pygame.transform.rotate(bisca_card.image, 90)
    bisca_card.rect = bisca_card.image.get_rect()

    bisca_pos_x = d_w * (1 - lm_p)
    bisca_pos_y = cards_sprites[1].rect.y - cs_p * c_w - c_h + 1/2*(c_h - c_w)
    bisca_card.rect.x = bisca_pos_x
    bisca_card.rect.y = bisca_pos_y

    table_card = cards_sprites[4]
    table_pos_x = d_w*(1 - rm_p) - c_w/2*(5 + 3*cs_p)
    table_pos_y = cards_sprites[1].rect.y - cs_p * c_w - c_h
    table_card.rect.x = table_pos_x
    table_card.rect.y = table_pos_y

    played_card = cards_sprites[5]
    played_pos_x = cards_sprites[4].rect.x + c_w*(1 + cs_p)
    played_pos_y = cards_sprites[4].rect.y
    played_card.rect.x = played_pos_x
    played_card.rect.y = played_pos_y


def main():
    ## Inicialize thing ###
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

    ### Object containing the information of my actions ###
    hand = Hand()
    HandCard.hand = hand
    #######

    ### Sprites for cards ###
    hand_sprites = [HandCard("CORINGA", screen, i) for i in range(3)]
    bisca_sprite = [BiscaCard("CORINGA", screen)]
    table_sprites = [Card("CORINGA", screen) for i in range(2)]
    cards_sprites = hand_sprites + bisca_sprite + table_sprites
    
    hand_group = pygame.sprite.Group(*hand_sprites)
    other_group = pygame.sprite.Group(*(bisca_sprite + table_sprites))
    ######

    ### Setup card's positions ###
    cs_p = 0.2
    c_ar = 0.64
    lm_p, rm_p = 1-0.25, 0.05  
    bm_p = 0.05

    position_setup = {"cs_p": cs_p, "c_ar":c_ar, "lm_p":lm_p, "rm_p":rm_p, 
                    "bm_p":bm_p, "game_display_shape":game_display_shape}

    update_cards_pos(cards_sprites, position_setup)
    ######

    ### Drawn cards ###
    hand_group.draw(screen)
    other_group.draw(screen)
    ###

    ### Bisca env ###
    # model to play with
    model_name = "bianca_v2"

    def show_observation(observation):
        tb_c = env.printable_card(observation[0])
        table_sprites[0].choose_image(tb_c)
        
        bisca_c = env.printable_card(observation[1])
        bisca_sprite[0].choose_image(bisca_c)
        
        for id, m_card in enumerate(observation[env.HAND_IDS]):
            h_card = env.printable_card(m_card)
            hand_sprites[id].choose_image(h_card)

    # Initial setup
    env = custom_env.Bisca()
    model_path = os.path.join("saved_model", model_name)
    model = load_model(model_path)

    first_move = random.randint(0,1)
    observation = env.reset(first_move=first_move, model=model)
    hand.reset()

    # Show initial state
    show_observation(observation)
    table_sprites[1].choose_image("VAZIO")
    ######

    # Label to inform who won each round
    bianca_win_roud_label = font.render("Bianca ganhou!", 1, (255,255,0))
    you_win_roud_label = font.render("Você ganhou!", 1, (255,255,0))
    upper_left_win = (10, 10)

    # Label to inform the percentage of wins
    pc_vic_label = font.render(f"Porcentagem de vitória:", 1, (255,255,0))
    upper_left_pc_vic = (10, 10 + bianca_win_roud_label.get_height() + 4)
    screen.blit(pc_vic_label, upper_left_pc_vic)
    games_count = 0
    victory_count = 0

    pygame.display.flip()

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
                    if winner == 1:
                        w, h = you_win_roud_label.get_width(), you_win_roud_label.get_height()
                    else:
                        w, h = bianca_win_roud_label.get_width(), bianca_win_roud_label.get_height()
                    screen.fill(pygame.Color(0,0,0,0), pygame.Rect(*upper_left_win,w, h))
                    
                    # Update the table cards for the next round
                    m_card = env.printable_card(new_state[0])
                    table_sprites[0].choose_image(m_card)
                    table_sprites[1].choose_image("VAZIO")

        if not showing_results:
            if done:
                done = False
                
                # Check who won and store that information
                games_count += 1
                if env.count_round_win[0] > env.count_round_win[1]:
                    victory_count += 1
                
                # Reset bisca environment
                first_move = random.randint(0,1)
                observation = env.reset(first_move=first_move, model=model)
                hand.reset()

                # Update win percentage label 
                w, h = pc_vic_label.get_width(), pc_vic_label.get_height()  
                screen.fill(pygame.Color(0,0,0,0), pygame.Rect(*upper_left_pc_vic,w, h))

                pc_vic = victory_count/games_count*100
                pc_vic_label = font.render(f"Porcentagem de vitória: {pc_vic:.2f} %", 1, (255,255,0))
                screen.blit(pc_vic_label, upper_left_pc_vic)

                # Display initial observation
                show_observation(observation)
            
            # Check if a hand's card was clicked and notify that to "hand"
            hand_group.update(events_list)

        if hand.action_chosen:
            hand.action_chosen = False
            showing_results = True

            new_state, reward, done, info = env.step(hand.action, model)
            winner = info[1]
            played_cards = (info[0][1], info[0][0])

            # Display label of who won
            if winner == 1:
                screen.blit(you_win_roud_label, upper_left_win)
            else:
                screen.blit(bianca_win_roud_label, upper_left_win)

            # Display played cards
            for sprite, m_card in zip(table_sprites, played_cards):
                h_card = env.printable_card(m_card)
                sprite.choose_image(h_card)

        clock.tick(60)

if __name__=="__main__":
    main()