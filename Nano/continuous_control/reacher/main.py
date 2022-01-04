
import torch
import argparse
import numpy as np
from src.agents import Agent_DDPG, agent_train, Agent_D4PG, Agent_TD3, Agent_TD3_4
from src.utils import load_env, save_pickle, plot_scores_training_all, plot_time_all, plot_play_scores
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
        if args.mode == "training" and agent_type == "2": # agent 2 is 20 arms
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
        elif args.mode == "play" and agent_type == "2":
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
    num_episodes= 1000 # default number of episodes

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
    elif args.mode == "play" :
        algor = ''
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

        # list for save scores
        score_play=[]
        for episode in range(3):
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations
            score = np.zeros(n_agents)

            while True:
                actions = agent.act(states, 10000,add_noise=False)

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
