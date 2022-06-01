import pygame
import os

class Background(pygame.sprite.Sprite):
    def __init__(self, image_name, game_display_shape, rotate=False) -> None:
        super().__init__()

        image_path = os.path.join("images", image_name)
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = (0, 0)

        if rotate:
            self.image = pygame.transform.rotate(self.image, 90)
            self.rect = self.image.get_rect()

        ### Reshape the background to fit the display ###
        image_ar = self.rect.width/self.rect.height
        display_ar = game_display_shape[0] / game_display_shape[1]
        if image_ar < display_ar:
            scale_factor = game_display_shape[0] / self.rect.width
        else:
            scale_factor = game_display_shape[1] / self.rect.height

        new_shape = (self.rect.width * scale_factor, self.rect.height * scale_factor)
        self.image = pygame.transform.scale(self.image, new_shape)
        self.rect = self.image.get_rect()
        ######


    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)