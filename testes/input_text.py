import pygame as pg


COLOR_INACTIVE = pg.Color('lightskyblue3')
COLOR_ACTIVE = pg.Color('dodgerblue2')

class ScreenChanged:
    has_changed = True

class InputBox(pg.sprite.Sprite):
    font = None

    def __init__(self, x, y, w, h, text=''):
        super().__init__()

        self.init_width = w
        self.rect = pg.Rect(x, y, w, h)
        self.old_rect = self.rect.copy()
        self.color = COLOR_INACTIVE
        self.text = text
        self.txt_surface = InputBox.font.render(text, True, self.color)
        self.active = False
        self.has_changed = True

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
            self.has_changed = True
        
        if self.active:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    print(self.text)
                    self.text = ''
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = InputBox.font.render(self.text, True, self.color)

                width = max(self.init_width, self.txt_surface.get_width()+10)
                self.old_rect = self.rect.copy()
                self.rect.w = width

                self.has_changed = True

    def update(self):
        # Resize the box if the text is too long.
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        if self.has_changed:
            # Clean Inputtext
            screen.fill((0, 0, 0), self.old_rect)
            # Back ground
            screen.fill((30, 30, 30), self.rect)
            # Blit the text.
            screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
            # Blit the rect.
            pg.draw.rect(screen, self.color, self.rect, 2)
            
            self.has_changed = False
            ScreenChanged.has_changed = True


def main():
    pg.init()
    screen = pg.display.set_mode((640, 480))
    clock = pg.time.Clock()

    font = pg.font.Font(None, 32)
    InputBox.font = font
    
    input_box1 = InputBox(100, 100, 140, 32)
    input_box2 = InputBox(100, 300, 140, 32)
    input_boxes = pg.sprite.Group(input_box1, input_box2)

    done = False
    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            
            for box in input_boxes.sprites():
                box.handle_event(event)

        # for box in input_boxes:
        #     box.update()

        for box in input_boxes.sprites():
            box.draw(screen)

        if ScreenChanged.has_changed:
            pg.display.flip()
            ScreenChanged.has_changed = False

        clock.tick(60)

main()