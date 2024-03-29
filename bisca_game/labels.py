from main import pygame
from managers import ScreenManager

class Label(pygame.sprite.Sprite):
    screen_manager: ScreenManager

    def __init__(self, font: pygame.font.Font, upper_left_pos: tuple=(0,0), text: str="", color=(255, 255, 0)) -> None:
        super().__init__()

        self.font = font
        self.color = color
        self.text = text
        
        self.upper_left_pos = upper_left_pos
        self.rect = upper_left_pos

        print(color)
        self.image = font.render(text, 1, color) 

    def update_text(self, text):
        self.text = text
        self.image = self.font.render(text, 1 , self.color)

        Label.screen_manager.has_changed = True

    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)


class PcWinLabel(Label):
    def __init__(self, font: pygame.font.Font, upper_left_pos: tuple=(0, 0), text: str="", color=(255, 2555, 0)) -> None:
        super().__init__(font, upper_left_pos, text, color)

        self.games_count = 0
        self.victory_count = 0

    def update_text(self):
        self.text = f"Porcentagem de vitória: {self.victory_count/self.games_count*100:.2f} %"
        self.image = self.font.render(self.text, 1 , self.color)
        Label.screen_manager.has_changed = True


class WinnerLabel(Label):
    bianca_win_round_text = "Bianca ganhou!"
    you_win_round_text = "Você ganhou!"

    def __init__(self, font: pygame.font.Font, upper_left_pos: tuple=(0, 0), text: str="", color=(255, 2555, 0)) -> None:
        super().__init__(font, upper_left_pos, text, color)

        self.font = font
        self.color = color
        self.image = pygame.Surface((0,0))
        
        self.bianca_win_text = "Bianca ganhou!"
        self.bianca_win_round_label = font.render(WinnerLabel.bianca_win_round_text, 1, color)
        self.you_win_round_label = font.render(WinnerLabel.you_win_round_text, 1, color)

    def set_you_win_label(self, player_name):
        self.you_win_round_label = self.font.render(f"{player_name} ganhou!", 1, self.color)

    def update_winner_label(self, winner):
        if winner == "p1":
            self.text = WinnerLabel.you_win_round_text
            self.image = self.you_win_round_label
        elif winner == "p2":
            self.text = WinnerLabel.bianca_win_round_text
            self.image = self.bianca_win_round_label
        else:
            self.text = ""
            self.image = pygame.Surface((0,0))

        WinnerLabel.screen_manager.has_changed = True
        

class NumGames(Label):
    def update_text(self, num_games):
        self.text = f"Partidas jogadas: {num_games}"
        self.image = self.font.render(self.text, 1 , self.color)
        Label.screen_manager.has_changed = True