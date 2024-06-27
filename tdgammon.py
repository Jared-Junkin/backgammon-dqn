
import gym
import copy
import os
import io
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from random import randint
from gym.spaces import Box
import logging
import torch
import ELO
from typing import List
from contextlib import redirect_stdout

class epsilonGreedy():
    def __init__(self, epsilon: float, epsilon_min_1: float, epsilon_min_2: float, epsilon_min_3: float, episode_len: float) -> None:
        self.epsilon = epsilon
        self.epsilon_max_1=epsilon
        self.epsilon_min_1=epsilon_min_1
        self.epsilon_max_2=epsilon_min_1
        self.epsilon_min_2=epsilon_min_2
        self.epsilon_max_3=epsilon_min_2
        self.epsilon_min_3=epsilon_min_3

        self.epsilon_interval_1 = (self.epsilon_max_1 - self.epsilon_min_1)
        self.epsilon_interval_2 = (self.epsilon_max_2 - self.epsilon_min_2)
        self.epsilon_interval_3 = (self.epsilon_max_3 - self.epsilon_min_3)

        self.episode_len = episode_len

    def calc_and_update_epsilon(self, steps: int) -> float:
        if steps <= 1*self.episode_len:
            self.epsilon -= self.epsilon_interval_1 / self.episode_len
            self.epsilon = max(self.epsilon, self.epsilon_min_1)
        elif steps <= 2*self.episode_len:
            self.epsilon -= self.epsilon_interval_2 / self.episode_len
            self.epsilon = max(self.epsilon, self.epsilon_min_2)
        elif steps <= 3*self.episode_len:
            self.epsilon -= self.epsilon_interval_3 / self.episode_len
            self.epsilon = max(self.epsilon, self.epsilon_min_3)
        else: 
            self.epsilon=0
        return self.epsilon
    
    def eps(self) -> float:
        return self.epsilon

seed = 42
random.seed(seed)

MAX_GAME_MOVES = 1000
RANDOMAGENT_ELO_SCORE = 100
# Set up logging
logging.basicConfig(filename='DQN_training.log', 
                    filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)

sys.path.insert(0, "/usr/src/app/gym-backgammon")
from gym_backgammon.envs.rendering import Viewer # would love to figure out how to get this to work somehow
from gym_backgammon.envs.backgammon import Backgammon as Game, WHITE, BLACK, COLORS, TOKEN


STATE_W = 96
STATE_H = 96

SCREEN_W = 600
SCREEN_H = 500


class RandomAgent:
    def __init__(self, color):
        self.color = color
        self.name = 'AgentExample({})'.format(self.color)

    def roll_dice(self):
        return (-random.randint(1, 6), -random.randint(1, 6)) if self.color == WHITE else (random.randint(1, 6), random.randint(1, 6))

    def choose_best_action(self, actions, env,  NN, device):
        return random.choice(list(actions)) if actions else None
    
    def fold(self, featurevec_full: list) -> list:
        if self.color == WHITE:
            return featurevec_full[:196]
        else:
            white_points = featurevec_full[:96]
            white_bar_off = featurevec_full[96:98]
            black_points = featurevec_full[98:194]
            black_bar_off = featurevec_full[194:196]

            reversed_black_points = []
            reversed_white_points = []
            for i in range(24, -1, -1):
                reversed_black_points.extend(black_points[i*4: (i+1)*4])
                reversed_white_points.extend(white_points[i*4: (i+1)*4])

            folded_board = reversed_black_points + black_bar_off + reversed_white_points + white_bar_off
            return folded_board


class TDGammonAgent(RandomAgent):
    def __init__(self, color: str, gamma: float = 0.9, lambda_: float = 0.8, learning_rate: float=1e-3, epsilon: float = 0.1) -> None:
        super().__init__(color)
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambda_ = lambda_ 
        self.learning_rate = learning_rate
    def TDError(self, V_current: float, V_next: float, reward_next: float) -> float:
        return (reward_next + self.gamma * V_next) - V_current

    # this calcluates the estimated value function V(s_t) 
    # (this function estimates the probability that the agent will win the game. the neural net NN learns to calculate this based on the state of the board)
    def estimateValueFunction(self, NN: torch.nn.modules.container.Sequential, featurevector: np.ndarray, device: str) -> float:
        vec = torch.tensor(featurevector, dtype=torch.float32).unsqueeze(0)     # Convert to pytorch tensor and add batch dimension 
        # print(vec.size())
        vec = vec.to(device)                                               # Move tensor to same device as model
        return NN(vec)                                                          # pass through neural net
    
    def choose_best_action(self, actions, env, NN, device): # env = full game environment
        eps = random.uniform(0, 1)
        if eps <= self.epsilon: # exploration policy
            return random.choice(list(actions)) if actions else None

        # save old state 
        state = env.game.save_state()
        code = self.color == "WHITE"
        best_action = None
        V_s_max = float("-inf")

        for action in actions:

            observation, reward, done, winner = env.step(action=action)
            featurevec = env.game.get_board_features(code)
            featurevec = self.fold(featurevec_full=featurevec)
            V_s = self.estimateValueFunction(NN=NN, featurevector=featurevec, device=device)
            if V_s > V_s_max:
                # print(f"V_s = {V_s}, V_s_max = {V_s_max}")
                V_s_max = V_s
                best_action = action

            env.game.restore_state(state)
        return best_action

    
    def roll_dice(self):
        return super().roll_dice()
    def fold(self, featurevec_full: list) -> list:
        return super().fold(featurevec_full)
    def choose_random_action(self, actions, env,  NN, device):
        return random.choice(list(actions)) if actions else None

class BackgammonEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'state_pixels']}

    def __init__(self):
        self.game = Game()
        self.current_agent = None

        low = np.zeros((198, 1))
        high = np.ones((198, 1))

        for i in range(3, 97, 4):
            high[i] = 6.0
        high[96] = 7.5

        for i in range(101, 195, 4):
            high[i] = 6.0
        high[194] = 7.5

        self.observation_space = Box(low=low, high=high)
        self.counter = 0
        self.max_length_episode = 10000
        self.viewer = None

    def step(self, action):
        self.game.execute_play(self.current_agent, action)

        # get the board representation from the opponent player perspective (the current player has already performed the move)
        observation = self.game.get_board_features(self.game.get_opponent(self.current_agent))

        reward = 0
        done = False

        winner = self.game.get_winner()

        if winner is not None or self.counter > self.max_length_episode:
            # practical-issues-in-temporal-difference-learning, pag.3
            # ...leading to a final reward signal z. In the simplest case, z = 1 if White wins and z = 0 if Black wins
            if winner == WHITE:
                reward = 1
            done = True

        self.counter += 1

        return observation, reward, done, winner

    def reset(self):
        # roll the dice
        roll = randint(1, 6), randint(1, 6)

        # roll the dice until they are different
        while roll[0] == roll[1]:
            roll = randint(1, 6), randint(1, 6)

        # set the current agent
        if roll[0] > roll[1]:
            self.current_agent = WHITE
            roll = (-roll[0], -roll[1])
        else:
            self.current_agent = BLACK

        self.game = Game()
        self.counter = 0

        return self.current_agent, roll, self.game.get_board_features(self.current_agent)

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array', 'state_pixels'], print(mode)

        if mode == 'human':
            self.game.render()
            return True
        else:
            if self.viewer is None:
                self.viewer = Viewer(SCREEN_W, SCREEN_H)

            if mode == 'rgb_array':
                width = SCREEN_W
                height = SCREEN_H

            else:
                assert mode == 'state_pixels', print(mode)
                width = STATE_W
                height = STATE_H

            return self.viewer.render(board=self.game.board, bar=self.game.bar, off=self.game.off, state_w=width, state_h=height)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_valid_actions(self, roll):
        return self.game.get_valid_plays(self.current_agent, roll)

    def get_opponent_agent(self):
        self.current_agent = self.game.get_opponent(self.current_agent)
        return self.current_agent

def fold_unit_test(env: BackgammonEnv, boardfeatures_before: list, boardfeatures_after: list) -> bool: # assumes agent = 1
    self_features = boardfeatures_before[:96]
    self_features_after = boardfeatures_after[:]

def one_ply(agent: TDGammonAgent, env: BackgammonEnv, MLP_agent: torch.nn.Sequential, device: str, choose_random: bool = False):
    roll = agent.roll_dice()
    actions = env.get_valid_actions(roll=roll)
    if choose_random:
        action = agent.choose_random_action(actions=actions, env=env, NN=MLP_agent, device=device)
    else:
        action = agent.choose_best_action(actions=actions, env=env, NN=MLP_agent, device=device)
    boardfeatures, reward, done, winner = env.step(action=action)
    return env, boardfeatures, reward, done, winner, action

def play_one_game(MLP_agent: torch.nn.Sequential, device: str, agents: List[TDGammonAgent], learning:bool = True, log: bool = True, verbose: bool = False, fig_name: str=None) -> int:
    white_certainty = []
    black_certainty = []
    env = BackgammonEnv()
    if learning:
        eligibility_traces = {id(param): torch.zeros_like(param.data) for param in MLP_agent.parameters()}

    agent_color, roll, observation = env.reset()
    done = False
    winner = None
    i=0

    while not done or winner is None:
        if i > MAX_GAME_MOVES: # call it a draw after each player has moved 500 times. # this happens so rarely that many backgammon players never encounter it
            logging.info("we have a tie")
            return 0.5
        i+=1
        env, boardfeatures_returned, reward, done, winner = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent) # take a step as the current agent
        boardfeatures = agents[agent_color].fold(boardfeatures_returned) # forward to input to NN is from the perspective of the current agent, whether black or white
        if learning:
            # estimate probability of winning given current state
            V_t = agents[agent_color].estimateValueFunction(NN=MLP_agent, featurevector=boardfeatures) 
            if agent_color == WHITE:
                white_certainty.append(V_t.item())
            elif agent_color == BLACK:
                black_certainty.append(V_t.item())
            state = copy.deepcopy(env.game.save_state())

            # switch to opponent agent and calculate their next move
            agent_color = env.get_opponent_agent()
            env, observation, reward_opp, done, winner_opp = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent)

            # switch back to original agent, calculate probability of original agent moving, given state resulting from opponents move
            agent_color = env.get_opponent_agent()
            fold = agents[agent_color].fold(env.game.get_board_features(agent_color))
            V_t_plus_1 = agents[agent_color].estimateValueFunction(NN=MLP_agent, featurevector=fold)
            # if env2.game.get_winner() is not None:
            #     print("hello, there.")

            if done and winner is not None: 
                # the issue here is that I was setting "while not done AND not winner" (if one of these conditions is met, it terminates and gives a reward of 1 (for victory)). 
                # However, "done" is set to true also when the opponent simply cannot move for a given turn, the result is that the agent was learning to also prefer situations where it gets stuck somewhere and can't move
                reward = 1
            # calculate td error
            TD_error = agents[agent_color].TDError(V_current=V_t, V_next=V_t_plus_1, reward_next=reward)
            loss = TD_error.pow(2)
            loss.backward()
            with torch.no_grad():
                for param in MLP_agent.parameters():
                    trace = eligibility_traces[id(param)]

                    trace.mul_(agents[agent_color].gamma * agents[agent_color].lambda_).add_(param.grad)

                    # Update the weights using the eligibility trace
                    param.sub_(-1*agents[agent_color].learning_rate * TD_error.squeeze() * trace)

            # switch to opponent agent for next ply
            agent_color = env.get_opponent_agent()
            # restore game state so opponent can make move
            env.game.restore_state(state)

        else: # if not learning, we just need to switch to the other opponent and that's it
            agent_color = env.get_opponent_agent()

        # I think that we shouldn't need to change boardfeatures back because it gets recalculated on the first line of the while loop from the perspective of the new agent.
    if log:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            env.game.render()
        final_board = buffer.getvalue()
        logging.info(f"\n{final_board}")
        logging.info(f"winner after {i//2} turns is {winner}")
    if verbose:
        env.game.render()
        print(f"winner after {i//2} turns is {winner}")

    env.close()  

        
    # Assuming black_certainty and white_certainty are lists or numpy arrays
    # Example: black_certainty = [0.5, 0.6, 0.7, ...], white_certainty = [0.4, 0.5, 0.6, ...]

    if fig_name:
        if len(black_certainty) > len(white_certainty):
            black_certainty = black_certainty[:len(white_certainty)]
        elif len(white_certainty) > len(black_certainty):
            white_certainty = white_certainty[:len(black_certainty)]
        turns = range(len(black_certainty))
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(turns, black_certainty, label=f'Black ({BLACK})')
        plt.plot(turns, white_certainty, label=f'White ({WHITE})')


        # Adding labels and title
        plt.xlabel('Turn')
        plt.ylabel('Certainty of Winning')
        plt.title(f'AI Win Prob. Winner is {winner}')
        plt.legend()

        # Save the figure
        plt.savefig(fig_name)

        # Optionally display the plot
        # plt.show()
        plt.close()


    return winner

def play_one_game_eval(MLP_agents: List[torch.nn.Sequential], device: str, agents: List[TDGammonAgent], learning: bool = True, log: bool = True, verbose: bool = False):
# def play_one_game(MLP_agent: torch.nn.Sequential, device: str, agents: List[TDGammonAgent], learning: bool = True, log: bool = True, verbose: bool = False):
    env = BackgammonEnv()
    if learning: 
        eligibility_traces = {param: torch.zeros_like(param.data) for param in MLP_agent.parameters()}
    agent_color, roll, observation = env.reset()
    done = False
    winner = None
    i=0
    while not done or winner is None: # while the most recent play was able to move and there is no winner
        if i > MAX_GAME_MOVES: # call it a draw after each player has moved 500 times. # this happens so rarely that many backgammon players never encounter it
            return 0.5
        i+=1

        env, boardfeatures, reward, done, winner = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agents[agent_color], device=device)
        # env, boardfeatures, reward, done, winner = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent, device=device)
        boardfeatures = agents[agent_color].fold(boardfeatures)

        agent_color = env.get_opponent_agent()

    if log:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            env.game.render()
        final_board = buffer.getvalue()
        logging.info(f"\n{final_board}")
        logging.info(f"winner after {i//2} turns is {winner}")
    if verbose:
        env.game.render()
        print(f"winner after {i//2} turns is {winner}")

    env.close()  
    return winner

def load_agent(agent: str, device: str) -> torch.nn.Sequential:
    if not os.path.isfile(agent):
        raise Exception(f"file {agent} not found")
    MLP_agent = torch.nn.Sequential(
        torch.nn.Linear(196, 80, device=device),
        torch.nn.ReLU(),
        torch.nn.Linear(80, 80, device=device),
        torch.nn.ReLU(),
        torch.nn.Linear(80, 1, device=device),
        torch.nn.Sigmoid()
    )
    checkpoint = torch.load(agent, device)
    MLP_agent.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(MLP_agent.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    MLP_agent.to(device)  # Ensure the model is on the right device
    MLP_agent.eval()
    return MLP_agent

def evaluate_models(agent1: str, agent2: str, num_games: int = 20) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents = [TDGammonAgent(color=WHITE, epsilon=0), TDGammonAgent(color=BLACK, epsilon=0)] # agent 0 = x = WHITE, agent 1 = o = BLACK.
    # print(f"loading agents[0] from {agent1}")
    MLP_1 = load_agent(agent=agent1, device=device)
    # print(f"loading agents[1] from {agent2}")
    MLP_2 = load_agent(agent=agent2, device=device)
    # print(f"agent[0] = x = white = {agent1}, agent, agent[1] = 0 = black = {agent2}")
    white_wins = 0
    black_wins = 0
    for i in range(num_games):
        winner = play_one_game_eval(MLP_agents=[MLP_1, MLP_2], device=device, agents=agents, learning=False, log = False, verbose=False)
        if winner == WHITE:
            white_wins += 1
        elif winner == BLACK:
            black_wins += 1
    for i in range(num_games):
        # print(f"starting game {num_games + i}")
        winner = play_one_game_eval(MLP_agents=[MLP_2, MLP_1], device=device, agents=agents, learning=False, log = False, verbose=False)
        if winner == WHITE:
            black_wins += 1
        elif winner == BLACK:
            white_wins += 1
    # print(f"Done. Played {num_games*2} games. agent1 ({agent1}) wins: {white_wins} agent2: ({agent2}) wins: {black_wins}")
    return white_wins, black_wins