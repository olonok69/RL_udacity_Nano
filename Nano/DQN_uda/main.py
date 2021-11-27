import gym
import matplotlib.pyplot as plt
from model import *
from dqn_agent import *
import argparse
from gym import wrappers

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display



def main():
    parser = argparse.ArgumentParser("DQN")


    parser.add_argument("--mode", type=str, help="training , play", required=True)
    parser.add_argument("--episodes", type=str, help="number episodes training" )
    parser.add_argument("--type", type=str, help="type 1-->DQN , type 2--> DQN PER, type 3--> Double DQN, "
                                                 "type 4-->Dueling DQN, type 5--> Rainbow DQN")

    args = parser.parse_args()
    # load environment
    # BipedalWalker-v2
    name= 'LunarLander'
    #name = 'MountainCar'
    version='v2'
    env = gym.make(f'{name}-{version}')
    typo = args.type

    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.shape)
    if args.type == "1":
        state_size = env.observation_space.shape[0]
        try:
            action_size = env.action_space.n
        except:
            action_size = env.action_space.shape[0]
        agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    elif args.type == "2":
        # parameters DQN with proritized Experience Replay
        num_frames = 20000
        memory_size = 2000
        batch_size = 64
        target_update = 100
        epsilon_decay = 1 / 2000
        agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)

    elif args.type == "3":
        # parameters Double DQN
        num_frames = 20000
        memory_size = 1000
        batch_size = 32
        target_update = 200
        epsilon_decay = 1 / 2000

        # train
        agent = DQNAgent_Double(env, memory_size, batch_size, target_update, epsilon_decay)
    # counter for versioning
    elif args.type == "4":
        # parameters Dueling DQN
        num_frames = 20000
        memory_size = 1000
        batch_size = 32
        target_update = 100
        epsilon_decay = 1 / 2000

        # train
        agent = DQNAgent_Dueling(env, memory_size, batch_size, target_update, epsilon_decay)

    elif args.type == "5":

        # parameters
        num_frames = 30000
        memory_size = 10000
        batch_size = 32
        target_update = 100

        # train
        agent = DQNAgent_Raimbow(env, memory_size, batch_size, target_update)

    with open('dqnv.txt') as f:
        build = str(f.readline())
    build = int(build) + 1

    if args.mode== "training" and args.type == "1":

        def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
            """Deep Q-Learning.

            Params
            ======
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            """
            scores = []  # list containing scores from each episode
            scores_window = deque(maxlen=100)  # last 100 scores
            eps = eps_start  # initialize epsilon
            i_episode=0
            for i_episode in range(1, n_episodes + 1):
                state = env.reset()
                score = 0
                for t in range(max_t):
                    action = agent.act(state, eps)
                    next_state, reward, done, _ = env.step(action)
                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        break
                scores_window.append(score)  # save most recent score
                scores.append(score)  # save most recent score
                eps = max(eps_end, eps_decay * eps)  # decrease epsilon
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                if np.mean(scores_window) >= 200.0 and name =="LunarLander":
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                                 np.mean(scores_window)))
                    torch.save(agent.qnetwork_local.state_dict(), f'./model/checkpoint_{name}_{typo}.pth')
                    break
                elif np.mean(scores_window) >= 300.0 and name == "BipedalWalker":
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                                 np.mean(
                                                                                                     scores_window)))
                    torch.save(agent.qnetwork_local.state_dict(), f'./model/checkpoint_{name}_{typo}.pth')
                    break
                elif (i_episode ) >= n_episodes and name == "MountainCar":
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                                 np.mean(
                                                                                                     scores_window)))
                    torch.save(agent.qnetwork_local.state_dict(), f'./model/checkpoint_{name}_{typo}.pth')
                    break
                    #MountainCar
            return scores, i_episode


        if args.episodes != None:
            scores, i_episode = dqn(n_episodes=int(args.episodes))
        else:
            scores, i_episode = dqn()

        # plot the scores
        fig = plt.figure(figsize=(16,12))

        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(f'./images/scores_.jpg')
    elif args.mode == "training" and args.type == "2": #DQN per
        scores= agent.train(num_frames)
        if np.mean(scores) >= 200.0:
            print('\nEnvironment solved in  episodes!\tAverage Score: {:.2f}'.format(np.mean(scores)))
            torch.save(agent.dqn.state_dict(), f'./model/checkpoint_{name}_{typo}.pth')
    elif args.mode == "training" and args.type == "3": # Double DQN
        scores= agent.train(num_frames)
        if np.mean(scores) >= 200.0:
            print('\nEnvironment solved in  episodes!\tAverage Score: {:.2f}'.format(np.mean(scores)))
            torch.save(agent.dqn.state_dict(), f'./model/checkpoint_{name}_{typo}.pth')
    elif args.mode == "training" and args.type == "4": # Dueling DQN
        scores= agent.train(num_frames)
        if np.mean(scores) >= 200.0:
            print('\nEnvironment solved in  episodes!\tAverage Score: {:.2f}'.format(np.mean(scores)))
            torch.save(agent.dqn.state_dict(), f'./model/checkpoint_{name}_{typo}.pth')

    elif args.mode == "training" and args.type == "5": # Raimbow DQN
        scores= agent.train(num_frames)
        if np.mean(scores) >= 200.0:
            print('\nEnvironment solved in  episodes!\tAverage Score: {:.2f}'.format(np.mean(scores)))
        torch.save(agent.dqn.state_dict(), f'./model/checkpoint_{name}_{typo}.pth')


    elif  args.mode== "play" and args.type == "1":
        # load the weights from file
        plt.ion()
        agent.qnetwork_local.load_state_dict(torch.load(f'./model/checkpoint_{name}_{typo}.pth'))

        for i in range(3):
            state = env.reset()
            for j in range(5000):
                action = agent.act(state)
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break

        env.close()
    elif  args.mode== "play" and args.type == "5":
        # load the weights from file
        plt.ion()
        agent.dqn.load_state_dict(torch.load(f'./model/checkpoint_{name}_{typo}.pth'))

        for i in range(3):
            state = env.reset()
            done = False
            score = 0
            for j in range(500):
                action = agent.select_action(state)
                env.render()
                state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break
            print("score: ", score)
        env.close()
    elif  args.mode== "video":
        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load(f'./checkpoint_{name}_{typo}.pth'))

        env = wrappers.Monitor(env, f"./tmp/{name}-v{build}")
        for i in range(1):
            state = env.reset()
            # img = plt.imshow(env.render(mode='rgb_array'))
            for j in range(500):
                action = agent.act(state)
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break

        env.close()
        with open('dqnv.txt', "r+") as f:
            data = f.read()
            f.seek(0)
            f.write(str(int(build)))
            f.truncate()
   # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()