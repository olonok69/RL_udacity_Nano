import argparse
from src.agents import *
from src.utils import *
import os.path
import pandas as pd
import pickle
import time
from src.hyper import hp_tuning

def main():
    # Command line Arguments
    parser = argparse.ArgumentParser("DQN")
    parser.add_argument("--mode", type=str, help="training , play, compare, complare_play, plot, hp_tuning",
                        required=True)
    parser.add_argument("--type", type=str, help="type 1-->Vanilla DQN , type 2--> Duelling DQN PBR, type 3--> Dueling"
                                                 "DQN, no PBR, type 4-->categorical DQN, type 5--> Duelling DQN"
                                                 " with Noisy layer and PBR, Type 6--> DQN n-steps, type 7 --> "
                                                 "Rainbow DQN", required=True)
    args = parser.parse_args()

    #load environment
    if args.mode != "compare" and args.mode != "compare_play" and args.mode != "plot" and args.mode != "hp_tuning":
        worker_id= 1
        base_port= 5005
        env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
    # serialice outputs
    fname = "monitor/output.csv"
    cols = ["Algo", "score", "episodes", "max_t", "PER", "mode", "eps_start", "eps_end", "eps_decay"]

    if os.path.isfile(fname):
        outcomes= pd.read_csv(fname)
    else:
        outcomes= pd.DataFrame(data=[], columns=cols)

    # default parameters
    eps_decay = 0.995  # decay multiplicator for epsilon greedy algo
    eps_end = 0.01  # epsilon greedy min threshold. Always leave the possibility to do exploration
    eps_start = 1.0  # epsilon greedy parameter start always in 1, first action always random
    max_t = 1000  # max number of steps per episode
    n_episodes = 2000  # max number of episodes on training phase

    algo = args.type# <-- type of algo to use for training

    if args.mode == "training" and algo == "1":  # DQN
        # create DQN agent
        agent = DQN_Agent(state_size=state_size, action_size=action_size, seed=0)
        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                        eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 0, "training",
                    eps_start, eps_end, eps_decay, fname)



        env.close()
    elif args.mode == "training" and algo == "2":  # Dueling DQN No Prioritary Buffer
        # create DQN agent
        agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=False)
        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                        eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 0, "training",
                    eps_start, eps_end, eps_decay, fname)
        env.close()
    elif args.mode == "training" and algo == "3":  # Dueling DQN with Prioritary Buffer
        # create DQN agent
        agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=True)
        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                        eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 1, "training",
                    eps_start, eps_end, eps_decay, fname)

        env.close()
    elif args.mode == "training" and algo == "4":  # Categorical DQN, No prioritary buffer
        # create DQN agent
        agent = Cat_DQN_Agent(state_size=state_size, action_size=action_size, seed=0, train=True)
        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                        eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 0, "training",
                    eps_start, eps_end, eps_decay, fname)

        env.close()
    elif args.mode == "training" and algo == "5":  # Dueling DQN, with Noisy and prioritary buffer
        # create DQN agent
        agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=True,
                 noisy=True,)
        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                        eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 1, "training",
                    eps_start, eps_end, eps_decay, fname)

        env.close()
    elif args.mode == "training" and algo == "6":  # DQN n-Steps
        # create DQN agent
        agent = DQN_N_steps_Agent(state_size=state_size, action_size=action_size, seed=0, n_step=3,
                                  train=True)
        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 0, "training",
                    eps_start, eps_end, eps_decay, fname)

        env.close()
    elif args.mode == "training" and algo == "7":  # Rainbow DQN n-Steps
        # create DQN agent NoisyNet + DuelingNet + Categorical DQN + Prority Experience Replay and N-steps
        agent = DQN_Rainbow_Agent(state_size=state_size, action_size=action_size, seed=0, n_step= 3,
                                  alpha = 0.2,beta = 0.6, prior_eps= 1e-6, v_min = 0.0,  v_max = 30.0,
                                  atom_size = 51, num_frames = n_episodes, train = True)


        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                        eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 1, "training",
                    eps_start, eps_end, eps_decay, fname)

        env.close()
    elif args.mode == "training" and algo == "8":  # Dueling DQN, with Noisy and NO prioritary buffer
        # create DQN agent
        agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=False,
                 noisy=True,)
        # train Agent
        scores, scores_window, epi, losses = dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                        eps_start, eps_end, eps_decay)
        # plot the scores
        plot_scores(scores, algo, epi)
        # plot the scores
        plot_losses(losses, algo, epi)
        # save scores
        save_scores(outcomes, algo, np.mean(scores_window), epi, max_t, 1, "training",
                    eps_start, eps_end, eps_decay, fname)

        env.close()
    elif args.mode == "play" :
        agent=None
        # create agent

        if algo == "1": #DQN
            agent = DQN_Agent(state_size=state_size, action_size=action_size, seed=11)
        elif algo == "2": # DDQN No Prioritary Buffer
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=11, prioritary_buffer=False)
        elif algo == "3": # DDQN with Prioritary Buffer
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=101, prioritary_buffer=True)
        elif algo == "4": # Categorical DQN, No prioritary buffer
            agent = Cat_DQN_Agent(state_size=state_size, action_size=action_size, seed=0, train=False)
        elif algo == "5":  # Dueling DQN, with Noisy with prioritary buffer
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=True,
                 noisy=True,)
        elif algo == "6": # DQN n-Steps
            agent = DQN_N_steps_Agent(state_size=state_size, action_size=action_size, seed=0, n_step=3,
                                      train=False)
        elif algo == "7": # Rainbow Agent
            agent = DQN_Rainbow_Agent(state_size=state_size, action_size=action_size, seed=0, n_step=3,
                                      alpha=0.2, beta=0.6, prior_eps=1e-6, v_min=0.0, v_max=30.0,
                                      atom_size=51, num_frames=n_episodes, train=True)
        elif algo == "8":
            # Dueling DQN, with Noisy NO prioritary buffer
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=False,
                               noisy=True, )
        # If object agent exists
        if agent != None:
            # load the weights from file
            agent.qnetwork_local.load_state_dict(torch.load(f'models\checkpoint_{algo}.pth'))
            # train_mode must be false, we are in Play mode
            env_info = env.reset(train_mode=False)[brain_name]  # reset the environment

            state = env_info.vector_observations[0]  # get the current state
            score = 0  # initialize the score to 0
            while True:
                action = agent.act(state)  # select an action
                action = action.astype(int) # just in case we get a float
                env_info = env.step(action)[brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
                if done:  # exit loop if episode finished
                    break

            print("Score: {}".format(score))

            # save scores
            save_scores(outcomes, algo, score, 0, max_t, 0, "play",
                        eps_start, eps_end, eps_decay, fname)
            env.close()

    elif args.mode == "compare_play":
        with open('outputs/scores.pickle', 'rb') as handle:
            outputs = pickle.load(handle)

        if str(algo) == "1": #DQN
            worker_id = 10
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = DQN_Agent(state_size=state_size, action_size=action_size, seed=11)
        elif algo == "2": # DDQN No Prioritary Buffer
            worker_id = 20
            base_port = 5006
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=11, prioritary_buffer=False)
        elif algo == "3": # DDQN with Prioritary Buffer
            worker_id = 30
            base_port = 5007
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=11, prioritary_buffer=True)
        elif algo == "4": # Categorical DQN, No prioritary buffer
            worker_id = 40
            base_port = 5008
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = Cat_DQN_Agent(state_size=state_size, action_size=action_size, seed=11, train=False)
        elif algo == "5":  # Dueling DQN, with Noisy with prioritary buffer
            worker_id = 50
            base_port = 5009
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=11, prioritary_buffer=True,
                 noisy=True,)
        elif algo == "6": # DQN n-Steps
            worker_id = 60
            base_port = 5016
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = DQN_N_steps_Agent(state_size=state_size, action_size=action_size, seed=0, n_step=3,
                                      train=False)
        elif algo == "7": # rainbow
            worker_id = 70
            base_port = 5017
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = DQN_Rainbow_Agent(state_size=state_size, action_size=action_size, seed=11, n_step=3,
                                      alpha=0.2, beta=0.6, prior_eps=1e-6, v_min=0.0, v_max=30.0,
                                      atom_size=51, num_frames=n_episodes, train=False)
        elif algo == "8":  # Dueling DQN, with Noisy and NO prioritary buffer
            worker_id = 80
            base_port = 5018
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=False,
                 noisy=True,)
        # If object agent exists
        if agent != None:
            # load the weights from file
            agent.qnetwork_local.load_state_dict(torch.load(f'models\checkpoint_{algo}.pth'))
            # train_mode must be false, we are in Play mode
            env_info = env.reset(train_mode=False)[brain_name]  # reset the environment

            state = env_info.vector_observations[0]  # get the current state
            score = 0  # initialize the score to 0
            while True:
                action = agent.act(state)  # select an action
                action = action.astype(int) # just in case we get a float
                env_info = env.step(action)[brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
                if done:  # exit loop if episode finished
                    break

            print("Score: {}".format(score))
            outputs[str(algo)]['play'] = score

            # save scores
            save_scores(outcomes, algo, score, 0, max_t, 0, "play",
                        eps_start, eps_end, eps_decay, fname)
            env.close()

            with open("outputs/scores.pickle", 'wb') as handle:
                pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.mode == "compare":
        with open('outputs/scores.pickle', 'rb') as handle:
            outputs = pickle.load(handle)

        print(f"Algo {algo}")
        if  str(algo) == "1":  # DQN
            time1=time.time()
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            outputs[str(algo)]={}
            # create DQN agent
            agent = DQN_Agent(state_size=state_size, action_size=action_size, seed=0)
            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores']=scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2-time1
            env.close()
            #time.sleep(90)
        elif str(algo) == "2":  # DDQN No Prioritary Buffer
            # load environment
            time1 = time.time()
            worker_id = 2
            base_port = 5006
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            outputs[str(algo)] = {}
            # create DQN agent
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=False)
            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores'] = scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2 - time1

            env.close()
            #time.sleep(90)
        elif str(algo) == "3":  # Dueling DQN with Prioritary Buffer
            time1 = time.time()
            # load environment
            worker_id = 3
            base_port = 5007
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            outputs[str(algo)] = {}
            # create DQN agent
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=True)
            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores'] = scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2 - time1

            env.close()
        elif str(algo) == "4":  # Categorical DQN, No prioritary buffer
            time1 = time.time()
            # load environment
            worker_id = 4
            base_port = 5008
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            outputs[str(algo)] = {}
            # create DQN agent
            agent = Cat_DQN_Agent(state_size=state_size, action_size=action_size, seed=0, train=True)
            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores'] = scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2 - time1

            env.close()
        elif str(algo) == "5":  # Dueling DQN, with Noisy with prioritary buffer
            time1 = time.time()
            # load environment
            worker_id = 5
            base_port = 5009
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)

            outputs[str(algo)] = {}
            # create DQN agent
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=True,
                               noisy=True, )
            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores'] = scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2 - time1
            env.close()
        elif str(algo) == "6":  # DQN n-Steps
            time1 = time.time()
            # load environment
            worker_id = 6
            base_port = 5010
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            outputs[str(algo)] = {}
            # create DQN agent
            agent = DQN_N_steps_Agent(state_size=state_size, action_size=action_size, seed=0, n_step=3,
                                      train=True)
            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores'] = scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2 - time1

            env.close()
        elif str(algo) == "7":  # Rainbow DQN n-Steps
            time1 = time.time()
            # load environment
            worker_id = 7
            base_port = 5011
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            outputs[str(algo)] = {}
            # create DQN agent
            agent = DQN_Rainbow_Agent(state_size=state_size, action_size=action_size, seed=0, n_step=3,
                                      alpha=0.2, beta=0.6, prior_eps=1e-6, v_min=0.0, v_max=30.0,
                                      atom_size=51, num_frames=n_episodes, train=True)

            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores'] = scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2 - time1

            env.close()
        elif str(algo) == "8":  # Dueling DQN, with Noisy and NO prioritary buffer
            time1 = time.time()
            # load environment
            worker_id = 8
            base_port = 5012
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port)
            outputs[str(algo)] = {}
            # create DQN agent
            agent = DDQN_Agent(state_size=state_size, action_size=action_size, seed=0, prioritary_buffer=False,
                 noisy=True,)

            # train Agent
            scores, scores_window, epi, losses = all_dqn_runner(env, brain_name, agent, algo, n_episodes, max_t,
                                                            eps_start, eps_end, eps_decay)
            outputs[str(algo)]['scores'] = scores
            outputs[str(algo)]['epi'] = epi
            outputs[str(algo)]['losses'] = losses
            time2 = time.time()
            outputs[str(algo)]['time'] = time2 - time1

            env.close()

        with open("outputs/scores.pickle" , 'wb') as handle:
            pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.mode == "plot":
        # plotting outcomes
        # plot score all algorithms after 2000 episodes
        labels = plot_scores_training_all()
        # plot loss all algorithms
        plot_losses_training_all()
        # plot max score after 2000 episodes all algorithms
        plot_play_scores(labels)
        # plot time to solve the environtment. 13 yellow bananas. all algorithms
        plot_time_all(labels)
        # plot number of episodes to solve the environtment. 13 yellow bananas. all algorithms
        plot_episodes_to_solve_all(labels)
    elif args.mode == "hp_tuning":
        # hyper parameter tuning DQN agent
        best_params, trials = hp_tuning("./env/Banana.exe")
        print (best_params)
        with open("outputs/trials.pickle" , 'wb') as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()