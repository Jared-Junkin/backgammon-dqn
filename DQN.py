from tdgammon import *
import util

logging.basicConfig(filename='DQN_training.log', 
                    filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)
PWD = '/usr/src/app/gym-backgammon/examples/'
TERMINAL_STATE_WON = [1 for _ in range(196)] # terminal state all agents transition to on having won
TERMINAL_STATE_LOST = [0 for _ in range(196)] # terminal state all agents transition to on having lost
num_games = 200000 # total number of training steps (this would be approximately 200,000 games of self-play)
max_memory_length = 190000 # max number of entries in all histories
epsilon_random_frames = 1000 # number of turns of purely random actions to open game. I don't think this is useful for backgammon
update_target_network = 1000 # update target network every 5000 turs
change_base_agent = update_target_network * 4 # change the agent the learning agent is playing against every 20000 turns
epsilon_greedy_frames = num_games // 8 # number of turns per exploration epoch
gamma = 0.99
model_save_name = "DQNprimary"
target_save_name = "DQNtarget"
eps = epsilonGreedy(epsilon=1.0,
                    epsilon_min_1=0.2,
                    epsilon_min_2=0.1,
                    epsilon_min_3=0.0,
                    episode_len=epsilon_greedy_frames)
update_after_actions = 20
batch_size = 64
action_history = []
state_history = []
state_next_history = []
rewards_history = []

episode_reward_history = []
running_reward = 0
episode_count = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agents = [TDGammonAgent(color=WHITE, gamma=gamma), TDGammonAgent(color=BLACK, gamma=gamma)]
MLP_agent = torch.nn.Sequential(
    torch.nn.Linear(196, 80),
    torch.nn.ReLU(),
    torch.nn.Linear(80, 80),
    torch.nn.ReLU(),
    torch.nn.Linear(80, 1),
    torch.nn.Sigmoid()
    # torch.nn.Tanh()
)
MLP_agent.to(device=device)
MLP_agent.eval()
model_target = torch.nn.Sequential(
    torch.nn.Linear(196, 80),
    torch.nn.ReLU(),
    torch.nn.Linear(80, 80),
    torch.nn.ReLU(),
    torch.nn.Linear(80, 1),
    # torch.nn.Tanh()
    torch.nn.Sigmoid()
)
model_target.to(device=device)
model_target.eval()
MLP_agent_opponent = torch.nn.Sequential(
    torch.nn.Linear(196, 80),
    torch.nn.ReLU(),
    torch.nn.Linear(80, 80),
    torch.nn.ReLU(),
    torch.nn.Linear(80, 1),
    torch.nn.Sigmoid()
    # torch.nn.Tanh()
)
MLP_agent_opponent.to(device=device)

optimizer = torch.optim.Adam(MLP_agent.parameters(), lr=1e-4)
target_optimizer = torch.optim.Adam(model_target.parameters(), lr=1e-4)
loss_function = torch.nn.MSELoss()
# loss_function = torch.nn.SmoothL1Loss()
plot_image_interval = 500

j=0

# terminal_win_example = None
# terminal_win_color = None
# terminal_win_env_example = None

agent_wins = 0
losses = []
V_t_win = []
V_t_loss = []

agent_running_mean_wins = []
first_win_boardfeatures = None
first_loss_boardfeatures = None
deltas = []
while j < num_games:
    if j > 0 and j % change_base_agent == 0:
        agent_running_mean_wins = []
        logging.info(f'loading in new base opponent agent at {mdl_name} after {j} games')
        MLP_agent_opponent = load_agent(agent=mdl_name, device=device)
        logging.info(f"successfully loaded in new agent.")
    i=0
    game_len=i
    agent_certainty = []
    env = BackgammonEnv()
    done = False
    winner = None

    episode_reward = 0
    # reset to starting positions
    agent_color, roll, observation = env.reset()
    agent_a_color = agent_color

    # if the nonlearning agent goes first
    if agent_color == 1:
        # make move as nonlearning agent
        env, observation, _, done, winner, action = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent_opponent, device=device)
        # switch to learning agent
        agent_color = env.get_opponent_agent() 
    learning_agent_color = agent_color

            # decay probability of taking random action (every full turn)
    if j%2 == 0:
        epsilon = eps.calc_and_update_epsilon(i)
        agents[0].epsilon = epsilon
        agents[1].epsilon = epsilon

    # do full game


    # once it's terminal, assign terminal reweard to all states from that game and add all to array
    while True:

        i += 1

        # before previous action
        state_history.append(agents[agent_color].fold(featurevec_full=observation)) # append old state to history

        # #learning agent moves 
        # if j < epsilon_random_frames: # if we're just starting out, choose randomly
        #     env, boardfeatures, _, done, winner, action = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent, device=device, choose_random=True)
        # else:
        env, boardfeatures, _, done, winner, action = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent, device=device)
        
        # append to appropriate arrays for memory replay
        action_history.append(action)


        # if the current agent has won
        if done and winner is not None:
            agent_wins += 1
            # reward=[1 for _ in range(i)]
            # rewards_history = rewards_history + reward
            if first_win_boardfeatures is None:
                first_win_boardfeatures = agents[0].fold(boardfeatures)
            rewards_history.append(1)
            state_next_history.append(TERMINAL_STATE_WON)
            # set up terminal state
            break


        # update parameters for plotting (agent A)
        V_t = agents[agent_color].estimateValueFunction(NN=MLP_agent, featurevector=agents[agent_color].fold(boardfeatures), device=device) 
        agent_certainty.append(V_t.item())


        #  nonlearning agent moves
        agent_color = env.get_opponent_agent()
        # opponent moves
        if j < 5000: # if we're just starting out, choose randomly
            env, boardfeatures, _, done, winner, action = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent_opponent, device=device, choose_random=True)
        else: # else, choose best action
            env, boardfeatures, _, done, winner, action = one_ply(agent=agents[agent_color], env=env, MLP_agent=MLP_agent_opponent, device=device)

        # if the opponent agent has won
        if done and winner is not None:
            # reward=[-1 for _ in range(i)]
            # rewards_history = rewards_history + reward
            rewards_history.append(-1)
            if first_loss_boardfeatures is None:
                first_loss_boardfeatures = agents[0].fold(boardfeatures)
            state_next_history.append(TERMINAL_STATE_LOST)
            # set up terminal state
            break

        # switch back to agent a for next turn
        agent_color = env.get_opponent_agent() 
 
        reward=0
        rewards_history.append(reward)
        state_next_history.append(agents[agent_color].fold(featurevec_full=boardfeatures))

        # rewards_history.append(reward)
        episode_reward += reward


        observation = boardfeatures # new old state = old new state
    
    # the game has ended now
    
    # update primary network every 20th action, once we have enough samples
    if j % update_after_actions == 0 and len(rewards_history) > batch_size:

        # sample uniformly and at random from past experiences
        indices = random.sample(range(len(rewards_history)), batch_size)

        # append to tensors for backpropagation
        sample_states=np.array([state_history[j] for j in indices])
        sample_next_states=np.array([state_next_history[j] for j in indices])
        sample_rewards=[rewards_history[j] for j in indices]
        sample_actions=[action_history[j] for j in indices]

        # estimate future rewards using target network
        # updated q values = (r + gammma * max_a' : Q(s', a'; theta -))
        # util.printnet(model_target)
        with torch.no_grad():
            input_tensor = torch.tensor(sample_next_states)
            input_tensor = input_tensor.to(device, dtype=torch.float32)

            rewards_tensor = torch.tensor(sample_rewards)
            rewards_tensor = rewards_tensor.to(device, dtype=torch.float32)
            future_rewards = model_target(input_tensor) # something isn't on cuda and should be and I don't have the energy to fix it right now.
        updated_q_values = rewards_tensor + gamma * torch.max(future_rewards, dim=1)[0]



        
        # util.printnet(model_target)

        # Q(s, a; theta)
        # util.printnet(MLP_agent)
        MLP_agent.train()
        input_tensor = torch.tensor(sample_states)
        input_tensor = input_tensor.to(device, dtype=torch.float32)
        present_estimated_rewards = MLP_agent(input_tensor).squeeze(dim=1)
        # print("###########################")
        # util.printnet(MLP_agent)
        # predicted Q values is Q(s, a; theta); target is (r + gammma * max_a' : Q(s', a'; theta -))
        loss = loss_function(present_estimated_rewards, updated_q_values)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        MLP_agent.eval()
        tmp = 0


        V_t = agents[agent_color].estimateValueFunction(NN=MLP_agent, featurevector=first_win_boardfeatures, device=device) 
        V_t_win.append(V_t.item())
        V_tl = agents[agent_color].estimateValueFunction(NN=MLP_agent, featurevector=first_loss_boardfeatures, device=device) 
        V_t_loss.append(V_tl.item())

        # print(f'real q values are: {updated_q_values}\nestimated q values are {present_estimated_rewards}\nloss is {loss}')
        # print("###########################")
        # util.printnet(MLP_agent)
        # util.printnet(model_target)
        # print("tests)")

    
    # update the network at the end of every 10000th game
    if j % update_target_network == 0:
        # save the old target and primary networks
        mdl_name = model_save_name + str(j) + ".pth"
        tgt_name = target_save_name + str(j) + ".pth"
        model_save_path = os.path.join(PWD, mdl_name)
        target_save_path = os.path.join(PWD, tgt_name)
        
        logging.info(f"saving MLP_AGENT at {model_save_path}")
        torch.save({
            'model_state_dict': MLP_agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)
        logging.info(f"saving target model at {target_save_path}")
        torch.save({
            'model_state_dict': model_target.state_dict(),
            'optimizer_state_dict': target_optimizer.state_dict(),
        }, target_save_path)
        # replace old target network with current primary network
        model_target.load_state_dict(MLP_agent.state_dict())

    

    if j % 500 == 0 and j > 0:
        plt.figure(figsize=(10,6))
        if len(V_t_win) > len(V_t_loss):
            V_t_win = V_t_win[:len(V_t_loss)]
        elif len(V_t_loss) > len(V_t_win):
            V_t_loss = V_t_loss[:len(V_t_loss)]
        turns = range(len(V_t_loss))
        plt.plot(turns, V_t_loss, label=f'Loss for agent')
        plt.plot(turns, V_t_win, label=f'win for agent')
                # Adding labels and title
        plt.xlabel('Turn')
        plt.ylabel('Certainty of Winning')
        plt.title(f'AI Win Prob for fixed boards with win and loss')
        plt.legend()

        # Save the figure
        plt.savefig("losswin.png")

        # Optionally display the plot
        # plt.show()
        plt.close()
    
        plt.figure(figsize=(10,6))
        turns = range(len(losses))
        plt.plot(turns, losses, label=f"value of loss function on game i")
        plt.xlabel('nth update')
        plt.ylabel('value of loss function')
        plt.title(f'value of loss function after n updates')
        plt.legend()

        # Save the figure
        plt.savefig("loss_fn.png")

        # Optionally display the plot
        # plt.show()
        plt.close()


    # remove excess entries from lists if they exceed max allowed size
    if len(rewards_history) > max_memory_length:
        del rewards_history[:1]
        del state_history[:1]
        del state_next_history[:1]
        del action_history[:1]
    
    if winner == learning_agent_color:
        deltas.append(agent_certainty[-1])
    if j % 20 == 0:
        agent_running_mean_wins.append(agent_wins/20)
        agent_wins = 0


    # save image every plot_image_interval games
    if j % plot_image_interval == 0:



        turns = range(len(agent_certainty))
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(turns, agent_certainty, label=f'Agent color ({learning_agent_color})')


        # Adding labels and title
        plt.xlabel('Turn')
        plt.ylabel('Certainty of Winning')
        plt.title(f'AI Win Prob. Winner is {winner}')
        plt.legend()

        # Save the figure
        name = "certainty"+str(j)+".png"
        plt.savefig(os.path.join(PWD, name))

        # Optionally display the plot
        # plt.show()
        plt.close()

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            env.game.render()
        final_board = buffer.getvalue()
        logging.info(f"\n{final_board}")

        plt.figure(figsize=(10,6))
        intervals = range(len(agent_running_mean_wins))
        plt.plot(intervals, agent_running_mean_wins)
        plt.xlabel("interval #")
        plt.ylabel("percentage of games won by learning agent")
        plt.title("percentage of games won over the course of this epoch")
        name="performance"+str(j)+".png"
        plt.savefig(os.path.join(PWD, name))
        plt.close()

        plt.figure(figsize=(10,6))
        games = range(len(deltas))
        plt.plot(games, deltas)
        plt.xlabel('Turn')
        plt.ylabel('winner certainty - loser certainty')
        plt.title(f"Difference in estimated prob. of winning (winner is {winner}")
        name = "probs"+str(j)+".png"
        plt.savefig(os.path.join(PWD, name))
        plt.close()
    logging.info(f"done game {j}. length = {i-game_len}. winner = {winner}. learning agent = {learning_agent_color}")
    j += 1




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
    # MLP_agent.eval()
    return MLP_agent