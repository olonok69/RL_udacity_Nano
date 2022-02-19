from collections import deque
import torch
import argparse
import numpy as np
from src.agents import Agent_DDPG, Agent_D4PG, Agent_TD3, Agent_TD3_4, Agent_A2C, Agent_PPO, Agent_A2C_b, Agent_SAC
from src.agents import soft_actor_critic_agent
from src.agents import  agent_train, agent_train_ac2, agent_train_ppo, agent_train_sac, sac_train
from src.utils import load_env, save_pickle, plot_scores_training_all, plot_time_all, plot_play_scores
from src.utils import Environment, plot_scores, plot_losses, ReplayMemory
from src.hyper import hp_tuning
import time
import os
import pickle

def main():
    # Command line Arguments
    parser = argparse.ArgumentParser("CC")
    parser.add_argument("--mode", type=str, help="training , play,  plot, hp_tuning",
                        required=True)
    parser.add_argument("--type", type=str, help="type 1-->DDQN , type 2--> TD3, type 3--> Td3 4DQN min"
                                                 "type 4-->Td3 4DQN mean, type 5--> D4PG"
                                                 "Type 6--> Td3 4DQN median "
                                                 , required=True)
    parser.add_argument("--agent", type=str, help="type of agent: 1--> 1 or 2--> 20 arms",
                        required=True)
    args = parser.parse_args()
    # dictionary to serialize metrics Algos for reporting and plotting
    fname = "outputs/outcomes.pkl"
    if os.path.isfile(fname):
        with open(fname, 'rb') as handle:
            outputs = pickle.load(handle)
    else:
        outputs = {}


    algo = args.type  # <-- type of algo to use for training
    agent_type= args.agent
    # load environment
    if args.mode != "compare" and args.mode != "compare_play" and args.mode != "plot" and args.mode != "hp_tuning":
        if args.mode == "training" and agent_type == "2" and not(algo in ["8", "10"]): # agent 2 is 20 arms
            file_name = "./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                        base_port,file_name,True, True)
        elif args.mode == "training" and agent_type == "1": # agent 1 is 1 arm
            file_name = "./envs/Reacher_Windows_x86_One/Reacher_Windows_x86_64/Reacher.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  True, True)
        elif args.mode == "play" and agent_type == "1":
            file_name = "./envs/Reacher_Windows_x86_One/Reacher_Windows_x86_64/Reacher.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  False, False)
        elif args.mode == "play" and agent_type == "2" and algo != "8":
            file_name = "./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                                  base_port, file_name,

                                                                                                  False, False)

    # CPU/ GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:' , device)
    #default checkpoint folder. Only final models
    CHECKPOINT_FOLDER = './models/'
    # Hyper-parameters
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 128  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 5e-3  # for soft update of target parameters
    LR_ACTOR = 3e-4  # learning rate of the actor
    LR_CRITIC = 1e-3  # learning rate of the critic
    WEIGHT_DECAY = 0  # L2 weight decay
    EXPLORATION_NOISE = 0.1 # sigma Normal Noise distribution for exploration
    TARGET_POLICY_NOISE = 0.2 # sigma Normal Noise distribution for target Networks
    TARGET_POLICY_NOISE_CLIP = 0.5 # clip target gaussian noise value
    num_episodes= 10000 # default number of episodes

    if args.mode == "training" and algo =="1": #DDQN

        time1 = time.time()
        # rewrite default parameters
        TAU = 1e-3  # for soft update of target parameters
        LR_ACTOR = 1e-4  # learning rate of the actor
        LR_CRITIC = 1e-4  # learning rate of the critic

        agent = Agent_DDPG(
                        device,
                        state_size, n_agents, action_size, 4,
                        BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                        algo, CHECKPOINT_FOLDER, True,
                )


        scores, loss_actor, loss_critic, mode = agent_train(env,brain_name, agent, n_agents, algo,num_episodes )

        time2 = time.time()
        times = time2 - time1
        # save outcomes for plotting
        save_pickle(outputs, scores, loss_actor, loss_critic, mode, fname, algo, times)
        env.close()
    elif args.mode == "training" and algo == "3":  # TD3 4 Critics min estimate selection
        time1 = time.time()
        agent = Agent_TD3_4(
            device,
            state_size, n_agents, action_size, 4,
            BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
            algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
            True, int(1e4), CHECKPOINT_FOLDER, True, "min"
        )

        scores, loss_actor, loss_critic, mode = agent_train(env, brain_name, agent, n_agents, algo, num_episodes)

        time2 = time.time()
        times = time2 - time1
        # save outcomes for plotting
        save_pickle(outputs, scores, loss_actor, loss_critic, mode, fname, algo, times)
        env.close()
    elif args.mode == "training" and algo =="2": #TD3
        time1 = time.time()
        agent = Agent_TD3(
                        device,
                        state_size, n_agents, action_size, 4,
                        BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                        algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
                        True,  int(1e4),CHECKPOINT_FOLDER, True,
                )

        scores, loss_actor, loss_critic, mode = agent_train(env, brain_name, agent, n_agents, algo, num_episodes)

        time2 = time.time()
        times = time2 - time1
        # save outcomes for plotting
        save_pickle(outputs, scores, loss_actor, loss_critic, mode, fname, algo, times)
        env.close()
    elif args.mode == "training" and algo == "4":  # TD3 4 Critics mean estimate selection
        time1 = time.time()
        agent = Agent_TD3_4(
            device,
            state_size, n_agents, action_size, 4,
            BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
            algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
            True, int(1e4), CHECKPOINT_FOLDER, True, "mean"
        )

        scores, loss_actor, loss_critic, mode = agent_train(env, brain_name, agent, n_agents, algo, num_episodes)

        time2 = time.time()
        times = time2 - time1
        # save outcomes for plotting
        save_pickle(outputs, scores, loss_actor, loss_critic, mode, fname, algo, times)
        env.close()
    elif args.mode == "training" and algo == "5": #D4PG
        # Note , this part is not finished. Only works with agent type 1 and need refactoring

        time1 = time.time()
        # rewrite hyper-parameters
        TAU = 1e-3  # for soft update of target parameters
        LR_ACTOR = 1e-4  # learning rate of the actor
        LR_CRITIC = 1e-4  # learning rate of the critic
        Vmax = 10
        Vmin = -10
        N_ATOMS = 51
        N_step = 1
        UPDATE_EVERY = 15
        BATCH_SIZE = 64
        num_episodes= 10000 # increase number of episodes as most likely we will use more than 1000
        agent = Agent_D4PG(
            device,
            state_size, n_agents, action_size, 4,
            BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
            algo, True, Vmax, Vmin, N_ATOMS, N_step,UPDATE_EVERY,
            CHECKPOINT_FOLDER, True,)
        scores, loss_actor, loss_critic, mode = agent_train(env, brain_name, agent, n_agents, algo, num_episodes)
        time2 = time.time()
        times = time2 - time1
        # save outcomes for plotting
        save_pickle(outputs, scores, loss_actor, loss_critic, mode, fname, algo, times)
        env.close()
    elif args.mode == "training" and algo == "6":  # TD3 4 Critics median estimate selection
        time1 = time.time()
        agent = Agent_TD3_4(
            device,
            state_size, n_agents, action_size, 4,
            BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
            algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
            True, int(1e4), CHECKPOINT_FOLDER, True, "median"
        )

        scores, loss_actor, loss_critic, mode = agent_train(env, brain_name, agent, n_agents, algo, num_episodes)

        time2 = time.time()
        times = time2 - time1
        # save outcomes for plotting
        save_pickle(outputs, scores, loss_actor, loss_critic, mode, fname, algo, times)
        env.close()
    elif args.mode == 'training' and algo == "7": # AC2
        time1 = time.time()
        rollout_length = 1  # steps to sample before bootstraping
        lr = 1e-4  # learning rate
        lr_decay = .99  # learning rate decay rate
        gamma = .95 # reward discount rate
        value_loss_weight = 1.0  # strength of value loss
        gradient_clip = 5 # threshold of gradient clip that prevent exploding

        agent = Agent_A2C_b(state_size, action_size, n_agents, device, algo, rollout_length,
                            lr, lr_decay, gamma, value_loss_weight, gradient_clip,
                            checkpoint_folder = './checkpoints/', train = True, mode='a2c')

        scores, loss_actor,  mode = agent_train_ac2(env, brain_name, agent, n_agents, algo, num_episodes)
        time2 = time.time()
        times = time2 - time1
        save_pickle(outputs, scores, loss_actor, loss_actor, mode, fname, algo, times)
        env.close()
    elif args.mode == 'training' and algo == "8":  # PPO2

        hyper_params = {
            'memory_size': 20000,  # replay buffer size
            'batch_size': 64,  # sample batch size
            't_random': 3,  # random steps at start of trajectory
            't_max': 1000,  # trajectory length
            'num_epochs': 10,  # number of updates
            'c_vf': 0.5,  # coefficent for vf loss (c1)
            'c_entropy': 0.001,  # starting value for coefficent for entropy (c2)
            'epsilon': 0.2,  # starting value for clipping parameter
            'gae_param': 0.95,  # gae param (λ)
            'discount': .99,  # discount (γ)
            'curation_percentile': 0,  # percent of trajectory data to drop
            'gradient_clip': 5,  # gradient clip
        }
        experiment= False
        time1 = time.time()
        file_name = "./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
        mode = "PPO"
        with Environment(file_name=file_name, no_graphics=True) as env:
            agent = Agent_PPO(env, hyper_params)
            rewards_window = [deque(maxlen=100) for n in range(env.num_agents)]
            loss_game = []
            scores_game = []
            for i_episode in range(1, num_episodes + 1):
                losses = []
                env.reset(train_mode=True)
                rewards_total = 0
                while True:
                    rewards, done = agent.collect_trajectories()
                    rewards_total += rewards
                    loss = agent.update()
                    losses.append(loss)
                    if done:
                        break
                # Track rewards
                for i, udr in enumerate(rewards_total):
                    rewards_window[i].append(udr)

                loss_game.append(np.mean(losses))
                scores_game.append(np.mean(rewards_total))
                print('\rEpisode {}\tAverage Reward: {:.2f} \tloss: {:.2f}'.format(i_episode, np.mean(rewards_total) ,
                                                                                   np.mean(losses)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Reward: {:.2f} \tloss: {:.2f}'.format(i_episode,
                                                                                       np.mean(rewards_total),
                                                                                       np.mean(losses)))
                mean_window_rewards = np.mean(rewards_window, axis=1)
                if (mean_window_rewards >= 35.0).all():
                    print('\nEnvironment solved in {:d} episodes.  Average agent rewards: '
                          .format(i_episode), mean_window_rewards)
                    torch.save(agent.policy.state_dict(), f'models/checkpoint_{algo}.pth')
                    break
        time2 = time.time()
        times = time2 - time1
        # save outcomes for plotting
        scores = np.mean(rewards_total)
        save_pickle(outputs, scores_game, loss_game, loss_game, mode, fname, algo, times)
        plot_scores(scores_game, algo, i_episode, "")
        plot_losses(loss_game, algo, i_episode, "model", "")
    elif args.mode == 'training' and algo == "9":  # SAC
        time1 = time.time()
        LR = 1e-5
        GAMMA = 0.95
        BATCH_SIZE = 64
        BUFFER_SIZE = int(1e6)
        ALPHA = .1
        TAU = 0.005
        TARGET_UPDATE_INTERVAL = 1
        GRADIENT_STEPS = 1

        agent = Agent_SAC(
                 state_size,
                 action_size,
                 LR,
                 GAMMA,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 ALPHA,
                 TAU,
                 TARGET_UPDATE_INTERVAL,
                 GRADIENT_STEPS,
                 n_agents,
                 device,
                 algo,
                 checkpoint_folder='./checkpoints/',
                 train=True,
                 mode='sac'
                 )

        num_episodes= 10000
        scores = agent_train_sac(env, brain_name, agent, n_agents, algo, num_episodes, BATCH_SIZE)
        time2 = time.time()
        times = time2 - time1
        #save_pickle(outputs, scores, loss_actor, loss_actor, mode, fname, algo, times)
        env.close()
    elif args.mode == 'training' and algo == "10":
        experiment = False
        time1 = time.time()
        file_name = "./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
        mode = "SAC"
        batch_size = 256
        threshold = 30
        LEARNING_RATE = 0.0007
        eval = True  ##
        start_steps = 10000  ## Steps sampling random actions
        max_steps = 1000
        replay_size = 1000000  ## size of replay buffer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with Environment(file_name=file_name, no_graphics=True) as env:
            seed = 0
            np.random.seed(seed)
            #env.seed(seed)
            agent = soft_actor_critic_agent(env.state_size, env.action_space_size ,
                                            device=device, hidden_size=256, seed=seed,
                                            lr=LEARNING_RATE, gamma=0.99, tau=0.005, alpha=0.2)
            memory = ReplayMemory(seed, replay_size)
            print('device: ', device)
            print('state dim: ', env.state_size)
            print('action dim: ', env.action_space_size)
            print('leraning rate: ', LEARNING_RATE)
            print('brain_name: ', env.brain_name)
            brain_name = env.brain_name

            scores, avg_scores = sac_train(max_steps, threshold, env, start_steps, agent, memory, batch_size, brain_name)

    elif args.mode == "play" :
        algor = ''
        episodes= 3
        if algo == "1":#DDPG
            # test the trained agent
            agent = Agent_DDPG(
                device, state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                algo, CHECKPOINT_FOLDER, False
            )
            mode= "DDPG"
            algor = algo + "_" + mode
        elif algo == "3": #TD3 4 DQN min estimate selection

            agent = Agent_TD3_4(
                device,
                state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
                True, int(1e4), CHECKPOINT_FOLDER, False, "min"
            )
            mode = "min"
            algor = algo + "_" + mode
        elif algo == "4": #TD3 4 DQN mean estimate selection

            agent = Agent_TD3_4(
                device,
                state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
                True, int(1e4), CHECKPOINT_FOLDER, False, "mean"
            )
            mode = "mean"
            algor = algo + "_" + mode
        elif algo == "2": # TD3

            agent = Agent_TD3(device,
                state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
                True, int(1e4), CHECKPOINT_FOLDER, False)
            mode = "TD3"
            algor = algo + "_" + mode
        elif algo == "6": #TD3 4 DQN median estimate selection
            agent = Agent_TD3_4(
                device,
                state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                algo, EXPLORATION_NOISE, TARGET_POLICY_NOISE, TARGET_POLICY_NOISE_CLIP,
                True, int(1e4), CHECKPOINT_FOLDER, False, "median"
            )

            mode = "median"
            algor = algo + "_" + mode
        elif algo == "7": # A2C
            rollout_length = 1  # steps to sample before bootstraping
            lr = 1e-4  # learning rate
            lr_decay = .99  # learning rate decay rate
            gamma = .95  # reward discount rate
            value_loss_weight = 1.0  # strength of value loss
            gradient_clip = 5  # threshold of gradient clip that prevent exploding

            agent = Agent_A2C_b(state_size, action_size, n_agents, device, algo, rollout_length,
                                lr, lr_decay, gamma, value_loss_weight, gradient_clip,
                                checkpoint_folder='./model/', train=False, mode='a2c')
            mode = "a2c"
            algor = algo + "_" + mode
        elif algo == "8":
            hyper_params = {
                'memory_size': 20000,  # replay buffer size
                'batch_size': 64,  # sample batch size
                't_random': 3,  # random steps at start of trajectory
                't_max': 1000,  # trajectory length
                'num_epochs': 10,  # number of updates
                'c_vf': 0.5,  # coefficent for vf loss (c1)
                'c_entropy': 0.001,  # starting value for coefficent for entropy (c2)
                'epsilon': 0.2,  # starting value for clipping parameter
                'gae_param': 0.95,  # gae param (λ)
                'discount': .99,  # discount (γ)
                'curation_percentile': 0,  # percent of trajectory data to drop
                'gradient_clip': 5,  # gradient clip
            }
            experiment = False
            time1 = time.time()
            file_name = "./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
            mode = "PPO"
            with Environment(file_name=file_name, no_graphics=False) as env:
                agent = Agent_A2C_b(env, hyper_params)
                agent.policy.load_state_dict(torch.load(f'models/checkpoint_{algo}.pth'))

                mode = "PPO"
                algor = algo + "_" + mode

                score_play = []
                for episode in range(episodes):
                    env.reset(train_mode=False)
                    states = env.info.vector_observations
                    rewards = 0
                    while True:
                        dist, _ = agent.policy(states)
                        actions = dist.sample()
                        env.step(actions.detach().numpy())
                        next_states = env.info.vector_observations
                        dones = env.info.local_done
                        rewards += np.array(env.info.rewards)
                        states = next_states
                        if np.any(dones):
                            break
                    print('\rRewards: ', rewards)
                    score_play.append( np.mean(rewards))
            if len(score_play) > 0:
                outputs[str(algor)]['score_play'] = np.mean(score_play)
                with open(fname, 'wb') as handle:
                    pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return

        # list for save scores
        score_play=[]
        for episode in range(episodes):
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations
            score = np.zeros(n_agents)

            while True:
                if algo != "7":
                    actions = agent.act(states, 10000,add_noise=False)

                    env_info = env.step(actions)[brain_name]
                elif algo == "7":
                    actions = agent.act(states)  # select actions
                    actions = actions.detach().cpu().numpy()
                    env_info = env.step(actions)[brain_name]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                score += rewards
                states = next_states

                if np.any(dones):
                    break

            print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))
            score_play.append(np.mean(score))


        if len(score_play)> 0 :
            outputs[str(algor)]['score_play'] = np.mean(score_play)

            with open(fname, 'wb') as handle:
                pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        env.close()
        return
    elif args.mode == "plot": # plot general outcomes
        # plot scores , time and training losses
        labels = plot_scores_training_all()
        plot_time_all(labels)
        plot_play_scores(labels)
    elif args.mode == "hp_tuning": # hyper parameter tuning
        # hyper parameter tuning DDPG agent
        file_name = "./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
        best_params, trials = hp_tuning(file_name)
        print(best_params)
        with open("outputs/trials.pickle", 'wb') as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
