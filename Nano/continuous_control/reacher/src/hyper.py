from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from functools import partial
from src.utils import *
from src.agents import Agent_DDPG


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
EXPLORATION_NOISE = 0.1 # sigma Normal Noise distribution for exploration
TARGET_POLICY_NOISE = 0.2 # sigma Normal Noise distribution for target Networks
TARGET_POLICY_NOISE_CLIP = 0.5 # clip target gaussian noise value
num_episodes= 1000
# default parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FOLDER = './models/'
algo="1"
def evaluate_model(hyperopt_params, env):

    LR= hyperopt_params['lr']
    BATCH_SIZE= hyperopt_params['batch_size']
    GAMMA= hyperopt_params['gamma']
    brain_name= hyperopt_params['brain_name']
    state_size = hyperopt_params['state_size']
    action_size = hyperopt_params['action_size']
    n_agents = hyperopt_params['n_agents']

    agent = Agent_DDPG(DEVICE,
                        state_size, n_agents, action_size, 4,
                        BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, LR, WEIGHT_DECAY,
                        algo, CHECKPOINT_FOLDER, True)

    scores = []
    loss_actor = []
    loss_critic = []
    scores_window = deque(maxlen=100)
    n_episodes = num_episodes


    for episode in range(1, 100 + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state
        agent.reset()  # reset the agent noise
        score = np.zeros(n_agents)
        while True:
            actions = agent.act(states, episode)

            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            agent.step(states, actions, rewards, next_states, dones)

            score += rewards  # update the score

            states = next_states  # roll over the state to next time step

            if np.any(dones):  # exit loop if episode finished
                break
        scores.append(np.mean(score))

        loss_actor.append(np.mean(agent.losses_actor))
        loss_critic.append(np.mean(agent.losses_critic))
        scores_window.append(np.mean(score))
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f} \tLoss critic: \t{:.2f} '
              '\tloss Actor: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window), np.mean(loss_critic)
                                              , np.mean(loss_actor)), end="")


    reward = np.mean(scores_window)

    return {'loss': -reward, 'status': STATUS_OK}

def objective(params, env):
    output = evaluate_model(params, env)
    return {'loss': output['loss'] ,  'status': output['status']}

def hp_tuning(file):

    file_name="./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
    worker_id = 1
    base_port = 5005
    env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                          base_port, file, True,
                                                                                          True)

    # define search Space
    search_space = { 'gamma': hp.loguniform('gamma' ,np.log(0.9), np.log(0.99)),
                    'batch_size' : hp.choice('batch_size', [32,64, 128]),
                     'lr': hp.loguniform('lr',np.log(1e-4), np.log(15e-3)),
                     'brain_name' : brain_name,
                     'state_size' : state_size,
                     'action_size' : action_size,
                     'n_agents' : n_agents,
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