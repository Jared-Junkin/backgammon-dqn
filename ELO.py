
# calculates the ELO score for agent with elo score R, opponent agent score R_opponent. S = 1 if agent wins, 0.5 if draw, 0 if loss
import math
def calc_ELO(R_opponent: float, R: float, S: float, K: int = 32, calc_both: bool = False) -> tuple:
    def calc_E(R_opponent: float, R: float) -> float:
        return 1 / (1 + math.pow(10, (R_opponent - R) / 400))

    E = calc_E(R_opponent=R_opponent, R=R)
    
    # Calculate new ELO for the agent
    new_R = R + K * (S - E)

    # Calculate new ELO for the opponent
    # S for the opponent is 1 - S for the agent
    if calc_both:
        new_R_opponent = R_opponent + K * ((1 - S) - (1 - E))

        return new_R, new_R_opponent
    else:
        return new_R, R_opponent
# print(calc_ELO(R_opponent=100, R=100, S=1, calc_both=True)) # if they're equal and agent wins, agent's score increases
# print(calc_ELO(R_opponent=100, R=100, S=1, calc_both=False)) # if they're equal and agent wins, agent's score increases. but if we set calc_both to false it won't change opponent's ELO.
# print(calc_ELO(R_opponent=100, R=100, S=0.5, calc_both=True)) # if they're equal and there's a draw, score doesn't change