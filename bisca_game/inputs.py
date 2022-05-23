from main import pygame
from managers import ScreenManager

class InputBox(pygame.sprite.Sprite):
    screen_manager: ScreenManager

    COLOR_INACTIVE = pygame.Color('lightskyblue3')
    COLOR_ACTIVE = pygame.Color('dodgerblue2')

    def __init__(self, x, y, w, h, font: pygame.font.Font, text=''):
        super().__init__()
        self.active = True
        self.text_submitted = ""
        self.enter_pressed = False
        
        self.rect = pygame.Rect(x, y, w, h)
        self.init_width = w
        self.old_rect = self.rect.copy()

        self.font = font
        self.text = text
        self.color = InputBox.COLOR_ACTIVE if self.active else InputBox.COLOR_INACTIVE
        
        self.txt_surface = self.font.render(text, True, self.color)


    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = InputBox.COLOR_ACTIVE if self.active else InputBox.COLOR_INACTIVE
            InputBox.screen_manager.has_changed = True

        if self.active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # print(self.text)
                    self.text_submitted = self.text
                    self.enter_pressed = True
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = self.font.render(self.text, True, self.color)

                width = max(self.init_width, self.txt_surface.get_width()+10)
                self.old_rect = self.rect.copy()
                self.rect.w = width

                InputBox.screen_manager.has_changed = True

                if self.enter_pressed:
                    self.enter_pressed = False
                    return self.text_submitted
            
            return None



    def draw(self, screen):
        # Back ground
        screen.fill((30, 30, 30), self.rect)
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)