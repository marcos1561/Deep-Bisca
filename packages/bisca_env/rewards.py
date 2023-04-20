from bisca_env.bisca_components import Card

class Reward():
    @staticmethod
    def get_reward(card1: Card, card2: Card, has_won_round: bool, done: bool) -> float:
        '''
        Reward given the cards played, the result of the round and the result of the game.
        '''
        pass

class CardPoints(Reward):
    win_reward = 20
    result_to_factor = {"win": 1, "loss": -1, "drawn": 0, "not_finished": 0}
    round_result_to_factor = {True: 1, False: -1}

    @staticmethod
    def get_reward(p1_card: Card, p2_card: Card, has_won_round: bool, done: bool, scores: dict[str, int], bisca: Card = None) -> float:
        result = "not_finished"
        if done:
            if scores["p1"] > scores["p2"]:
                result = "win"
            elif scores["p1"] == scores["p2"]:
                result = "drawn"
            else:
                result = "loss"

        return 1/2*(p1_card.points + p2_card.points) * CardPoints.round_result_to_factor[has_won_round] + CardPoints.win_reward * CardPoints.result_to_factor[result]

class BiscaPunishment(Reward):
    win_reward = 20
    result_to_factor = {"win": 1, "loss": -1, "drawn": 0, "not_finished": 0}
    round_result_to_factor = {True: 1, False: -1}

    const_bisca_punishment = 3
    factor_bisca_punishment = 1


    @staticmethod
    def bisca_punishment(p1_card: Card, p2_card: Card, bisca: Card):
        if p1_card.suit != bisca.suit or p2_card.suit == bisca.suit:
            return 0
        
        diff = p1_card.points - p2_card.points
        return BiscaPunishment.const_bisca_punishment + diff * BiscaPunishment.factor_bisca_punishment

    @staticmethod
    def get_reward(p1_card: Card, p2_card: Card, has_won_round: bool, done: bool, scores: dict[str, int], bisca: Card) -> float:
        result = "not_finished"
        if done:
            if scores["p1"] > scores["p2"]:
                result = "win"
            elif scores["p1"] == scores["p2"]:
                result = "drawn"
            else:
                result = "loss"

        points_reward = 1/2*(p1_card.points + p2_card.points) * BiscaPunishment.round_result_to_factor[has_won_round]
        win_game_reward = BiscaPunishment.win_reward * BiscaPunishment.result_to_factor[result]
        bisca_punishment = BiscaPunishment.bisca_punishment(p1_card, p2_card, bisca)

        return points_reward + win_game_reward - bisca_punishment 

