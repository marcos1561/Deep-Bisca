from main import pygame

class ScreenManager:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.has_changed = True

    def update_screen(self, sprites, backgroud=None):
        if self.has_changed:
            self.screen.fill((0,0,0))

            for s in sprites:
                s.draw(self.screen)

            pygame.display.flip()
            self.has_changed = False

class SceneManager:
    def __init__(self) -> None:
        self.current_scene = 0

class HandManager:
    def reset(self):
        self.action = 0
        self.action_chosen = False
        self.hand_idx_to_action = {0:0 ,1:1, 2:2}    
