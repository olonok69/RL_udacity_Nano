from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from src.agents import *
from functools import partial

from src.utils import *
from src.agents import DQN_Agent


# default parameters
eps_decay = 0.995  # decay multiplicator for epsilon greedy algo
eps_end = 0.01  # epsilon greedy min threshold. Always leave the possibility to do exploration
eps_start = 1.0  # epsilon greedy parameter start always in 1, first action always random
max_t = 1000  # max number of steps per episode
n_episodes = 2000  # max number of episodes on training phase

def evaluate_model(hyperopt_params, env):

    LR= hyperopt_params['lr']
    BATCH_SIZE= hyperopt_params['batch_size']
    GAMMA= hyperopt_params['gamma']
    brain_name= hyperopt_params['brain_name']
    state_size = hyperopt_params['state_size']
    action_size = hyperopt_params['action_size']

    agent = DQN_Agent(state_size=state_size, action_size=action_size, seed=0, lr=LR,
                 gamma=GAMMA,
                 batch_size=BATCH_SIZE)
    scores = []  # list containing scores from each episode
    losses = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = 1.0  # initialize epsilon
    epi = 0  # number of episodes
    for i_episode in range(1, 500 + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        # do max_t steps for this episode
        for t in range(max_t):
            action = agent.act(state, eps)
            action = action.astype(int)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))  # save most recent score
        losses.append(np.mean(agent.losses))  # append losses for statistics
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))


    reward = np.mean(scores_window)

    return {'loss': -reward, 'status': STATUS_OK}

def objective(params, env):
    output = evaluate_model(params, env)
    return {'loss': output['loss'] ,  'status': output['status']}

def hp_tuning(file):
    worker_id= 1
    base_port= 5005
    env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id, base_port,
                                                                                file=file)

    # define search Space
    search_space = { 'gamma': hp.loguniform('gamma' ,np.log(0.9), np.log(0.99)),
                    'batch_size' : hp.choice('batch_size', [32,64, 128]),
                     'lr': hp.loguniform('lr',np.log(1e-4), np.log(15e-3)),
                     'brain_name' : brain_name,
                     'state_size' : state_size,
                     'action_size' : action_size,
                               }
    # send the env with partial as additional env
    fmin_objective = partial(objective, env=env)
    trials = Trials()
    argmin = fmin(
        fn=fmin_objective,
        space=search_space,
        algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
        max_evals=30,
        trials=trials,
        verbose=True
        )#
    # return the best parameters
    best_parms = space_eval(search_space, argmin)
    return best_parms, trials