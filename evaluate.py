from play import * # this also imports all the import packages
import ELO
import json


# Function to save a dictionary to a file
def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

# Function to load a dictionary from a file
def load_dict_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def play_one_game_human(MLP_agent: torch.nn.Sequential, device: str, agents: List, learning: bool = True, log: bool = True, verbose: bool = False) -> int:
    env = BackgammonEnv()
    agent_color, roll, observation = env.reset()
    done = False
    winner = None
    i=0
    # env.game.render()
    if type(agents[0] == TDGammonAgent):
        HUMAN_AGENT_INDEX = 1
        HUMAN_AGENT_CHAR = 'O'
        print(f"Starting game. you are {HUMAN_AGENT_CHAR}")
    else:
        HUMAN_AGENT_INDEX = 0
        HUMAN_AGENT_CHAR = 'X'
        print(f"you are {HUMAN_AGENT_CHAR}")
    while not done or winner is None:
        if i > MAX_GAME_MOVES: # call it a draw after each player has moved 500 times. # this happens so rarely that many backgammon players never encounter it
            return 0.5
        i+=1
        if agent_color == HUMAN_AGENT_INDEX: # YOUR TURN    
            env.game.render()
            print(f"Starting turn {i}. (you are {HUMAN_AGENT_CHAR}).")
            action = agents[agent_color].choose_best_action(env=env)
            boardfeatures, reward, done, winner = env.step(action=action)
            print(f"you have chosen action {action}. New board: ")
            # env.game.render()
        else: # human's turn
            print(F"Starting AI Move")
            env, boardfeatures, reward, done, winner = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent, device=device, human=True)


        agent_color = env.get_opponent_agent()

    print(F"done. Winner is {winner}")
    env.game.render()
    return winner 
def play_one_game_eval(MLP_agents: List[torch.nn.Sequential], device: str, agents: List[TDGammonAgent], learning: bool = True, log: bool = True, verbose: bool = False):
# def play_one_game(MLP_agent: torch.nn.Sequential, device: str, agents: List[TDGammonAgent], learning: bool = True, log: bool = True, verbose: bool = False):
    env = BackgammonEnv()
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

def evaluate_models(agent1: str, agent2: str, elo_self: int = 100, elo_opponent: int = 100, num_games: int = 20, calc_both: bool = True, elo: bool = False) -> None:
    device = "cpu"
    agents = [TDGammonAgent(color=WHITE), TDGammonAgent(color=BLACK)] # agent 0 = x = WHITE, agent 1 = o = BLACK.
    # print(f"loading agents[0] from {agent1}")
    MLP_1 = load_agent(agent=agent1, device=device)
    # print(f"loading agents[1] from {agent2}")
    MLP_2 = load_agent(agent=agent2, device=device)
    agents[0].epsilon = 0
    agents[1].epsilon = 0
    # print(f"agent[0] = x = white = {agent1}, agent, agent[1] = 0 = black = {agent2}")
    white_wins = 0
    black_wins = 0
    if elo:
        for i in range(num_games):
            winner = play_one_game_eval(MLP_agents=[MLP_1, MLP_2], device=device, agents=agents, learning=False, log = False, verbose=False)
            if winner == WHITE:
                white_wins += 1
                elo_self, elo_opponent = ELO.calc_ELO(R_opponent=elo_opponent, R=elo_self, S=1, calc_both=calc_both)
            elif winner == BLACK:
                elo_self, elo_opponent = ELO.calc_ELO(R_opponent=elo_opponent, R=elo_self, S=0, calc_both=calc_both)
                black_wins += 1
            # print(f"starting game {num_games + i}")
            winner = play_one_game_eval(MLP_agents=[MLP_2, MLP_1], device=device, agents=agents, learning=False, log = False, verbose=False)
            if winner == WHITE:
                elo_self, elo_opponent = ELO.calc_ELO(R_opponent=elo_opponent, R=elo_self, S=0, calc_both=calc_both)
                black_wins += 1
            elif winner == BLACK:
                elo_self, elo_opponent = ELO.calc_ELO(R_opponent=elo_opponent, R=elo_self, S=1, calc_both=calc_both)
                white_wins += 1
        # print(f"Done. Played {num_games*2} games. agent1 ({agent1}) wins: {white_wins} agent2: ({agent2}) wins: {black_wins}")
        return white_wins, black_wins, elo_self, elo_opponent
    else:
        for i in range(num_games):
            winner = play_one_game_eval(MLP_agents=[MLP_1, MLP_2], device=device, agents=agents, learning=False, log = False, verbose=False)
            if winner == WHITE:
                white_wins += 1
            elif winner == BLACK:
                black_wins += 1
            # print(f"starting game {num_games + i}")
            winner = play_one_game_eval(MLP_agents=[MLP_2, MLP_1], device=device, agents=agents, learning=False, log = False, verbose=False)
            if winner == WHITE:
                black_wins += 1
            elif winner == BLACK:
                white_wins += 1
    # print(f"Done. Played {num_games*2} games. agent1 ({agent1}) wins: {white_wins} agent2: ({agent2}) wins: {black_wins}")
    return white_wins, black_wins

def roundRobin(start: int, end: int, interval: int, pwd : str= "/usr/src/app/gym-backgammon/examples", cpkt_base: str = "DQNprimary100.pth", savename="EloScores.txt", elo_scores: dict = None, num_opponents=20) -> None:
    
    if elo_scores is None:
        elo_scores = dict()
        elo_scores[cpkt_base] = [100, 0, 0]

    # evaluate all models against base to get base elo score
    for i in range(end, start, -interval):
        ckpt1 = "DQNprimary" + str(i) + ".pth"
        if ckpt1 not in elo_scores.keys():
        #     agent0_elo = 100
        #     agent0_wins, agent1_wins, agent0_elo, agent1_elo = evaluate_models(agent1=os.path.join(pwd, ckpt1), agent2=os.path.join(pwd, cpkt_base), elo_self= agent0_elo, elo_opponent=100, num_games=50, calc_both=False, elo=True)
            elo_scores[ckpt1] = [100, 0, 0]
        #     print(f"evaluated model last. {cpkt_base} wins: {agent1_wins}, {ckpt1} wins: {agent0_wins}, {ckpt1} elo score: {agent0_elo}")
        #     print(f'agent {ckpt1}: ELO: {agent0_elo}, ({agent0_wins} - {agent1_wins}) against base')
    
    # print("Starting Round Robin Play")
    # for i in range(start, end, interval):
    #     ckpt1 = "DQNprimary" + str(i) + ".pth"
              

        opponents = [x for x in range(15000, end, interval) if x != i]
        # opponents = random.sample(possible_opponents, 20)
        # opponents = random.sample(range(start, end, interval), 1) # just deciding that they'll have 12 opponents each
        for opponent in opponents:
            ckpt2 = "DQNprimary" + str(opponent) + ".pth"
            if ckpt2 not in elo_scores.keys():
                elo_scores[ckpt2] = [100, 0, 0]
            agent0_wins, agent1_wins, agent0_elo, agent1_elo = evaluate_models(agent1=os.path.join(pwd, ckpt1), agent2=os.path.join(pwd, ckpt2), elo_self= elo_scores[ckpt1][0], elo_opponent=elo_scores[ckpt2][0], num_games=5, calc_both=True, elo=True)
            print(f"Agent {i} ({elo_scores[ckpt1][1]} - {elo_scores[ckpt1][2]}) vs Agent {opponent} ({elo_scores[ckpt2][1]} - {elo_scores[ckpt2][2]}): Agent {i} record: ({agent0_wins} - {agent1_wins}).")
            elo_scores[ckpt1][0] = agent0_elo    # agent0 elo
            elo_scores[ckpt1][1] += agent0_wins   # agent0 wins
            elo_scores[ckpt1][2] += agent1_wins   # agent0 losses
            elo_scores[ckpt2][0] = agent1_elo    # agent1 elo
            elo_scores[ckpt2][1] += agent1_wins   # agent1 wins
            elo_scores[ckpt2][2] += agent0_wins   # agent1 losses
            # print(f"overall agent {i} record: ({elo_scores[ckpt1][1]} - {elo_scores[ckpt1][2]}). Agent {opponent} record: ({elo_scores[ckpt2][1]} - {elo_scores[ckpt2][2]})")
        filename = "ELO_results.json"
        save_dict_to_file(dictionary=elo_scores, filename=filename)
        print(f"done agent {i}. saved results to dict")

if __name__ == "__main__":
    pwd = "/usr/src/app/gym-backgammon/examples"
    ckpt1 = "DQNprimary260000.pth"
    device = "cpu"
    agent1=os.path.join(pwd, ckpt1)
    agent = load_agent(agent=agent1, device=device)
    play_one_game_human(MLP_agent=agent,device=device, agents=[TDGammonAgent(color=WHITE), humanAgent(color=BLACK)])

    # pwd = "/usr/src/app/gym-backgammon/examples"
    # ckpt1 = "DQNprimary200000.pth"
    # ckpt2 = "DQNprimary100000.pth"
    # agent0_wins, agent1_wins = evaluate_models(agent1=os.path.join(pwd, ckpt1), agent2=os.path.join(pwd, ckpt2), num_games=50)
    # print(f"evaluated model last. {ckpt2} wins: {agent1_wins}, {ckpt1} wins: {agent0_wins}")



    # filename = "ELO_results.json"
    # roundRobin(start=1000, end=66000, interval=1000)
    # Load the dictionary back from the file
    # loaded_dict = load_dict_from_file(filename)
    # print(loaded_dict)
    # roundRobin(start=160000, end=295000, interval=5000)

    # wins = []
    # s = []
    # for i in range(1000, 38000, 1000):
    #     ckpt1 = "DQNprimary" + str(i) + ".pth"
    #     agent0_wins, agent1_wins = evaluate_models(agent1=os.path.join(pwd, ckpt1), agent2=os.path.join(pwd, ckpt2), num_games=50)
    #     print(f"evaluated model last. {ckpt2} wins: {agent1_wins}, {ckpt1} wins: {agent0_wins}")
    #     wins.append(agent0_wins)
    #     s.append(i)
    # plt.figure(figsize=(10,6))
    # plt.plot(s, wins)
    # plt.xlabel("i games of self play")
    # plt.ylabel("wins out of 100")
    # plt.title("Wins out of 100 against random agent")
    # plt.savefig("evaluation_2.png")
    # plt.close()
   