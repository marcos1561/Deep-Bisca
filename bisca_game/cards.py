from main import pygame
from managers import ScreenManager, HandManager
import os

import bisca_components as bisca_components

class Card(pygame.sprite.Sprite):
    screen_manager: ScreenManager

    def __init__(self, card_name):
        super().__init__()
  
        self.name = card_name
        
        self.image = Card.load_card_image(card_name)
        self.rect = self.image.get_rect()

    @staticmethod
    def load_card_image(card_name):
        card_path = os.path.join("images", "cards_images", card_name + ".png")
        card_image = pygame.image.load(card_path).convert_alpha()
        return card_image

    def update_image(self, card_name):
        self.name = card_name
        if card_name == "VAZIO":
            self.image.set_alpha(0)
        else:
            self.image.set_alpha(255)
            self.image = Card.load_card_image(card_name)
            self.image = pygame.transform.scale(self.image, (self.rect.width, self.rect.height))

        Card.screen_manager.has_changed = True

    @staticmethod
    def card_name_from_bisca_card(bisca_card: bisca_components.Card):
        if bisca_card.is_null:
            return "VAZIO"

        bisca_suit_to_suit_name = {0: "C", 1: "E", 2: "P", 3: "O"}
        return f"{bisca_card.number}-{bisca_suit_to_suit_name[bisca_card.suit]}"

class BiscaCard(Card):
    def update_image(self, card_name):
        if card_name == "VAZIO":
            self.image.fill(pygame.Color(0,0,0,0))
        else:
            self.image = Card.load_card_image(card_name)
            self.image = pygame.transform.rotate(self.image, 90)
            self.image = pygame.transform.scale(self.image, (self.rect.width, self.rect.height))
        
        self.screen_manager.has_changed = True

class HandCard(Card):
    hand: HandManager

    def __init__(self, card_name, card_idx):
        super().__init__(card_name)

        self.card_idx = card_idx
        self.was_clicked = False


    def update(self, events_list) -> None:
        if self.name == "VAZIO":
            return

        selected = self.was_selected(events_list)
        if selected:
            self.update_image("VAZIO")
            
            # HandCard.hand.action = HandCard.hand.hand_idx_to_action[self.card_idx]
            HandCard.hand.action = self.card_idx
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