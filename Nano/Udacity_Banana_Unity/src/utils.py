import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import pickle

def plot_scores(scores , algo, num_episodes):
    """
    plot scores and save to disk
    :param scores: list of scores during training
    :param algo: type algorithm
    :return:
    """
    # plot the scores
    text = ""
    if algo == "1":
        text= "DQN Agent"
    elif algo == "3":
        text = "Dueling DQN Agent with Priority Buffer"
    elif algo == "4":
        text = "Categorical DQN Agent No prioritary buffer"
    elif algo == "2":
        text = "Dueling DQN Agent"
    elif algo == "5":
        text = "Dueling Noisy DQN Agent with Priority Buffer"
    elif algo == "6":
        text = "DQN n-Steps Agent"
    elif algo == "7":
        text = "DQN Rainbow Agent"
    elif algo == "8":
        text = "Dueling Noisy DQN Agent No PER"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'Algo {text} Number episodes {num_episodes}')
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'./images/scores_{algo}.jpg')
    return

def plot_losses(losses , algo, num_episodes):
    """
    plot scores and save to disk
    :param scores: list of scores during training
    :param algo: type algorithm
    :return:
    """
    # plot the scores
    text = ""
    if algo == "1":
        text= "DQN Agent"
    elif algo == "3":
        text = "Dueling DQN Agent with Priority Buffer"
    elif algo == "4":
        text = "Categorical DQN Agent No prioritary buffer"
    elif algo == "2":
        text = "Dueling DQN Agent"
    elif algo == "5":
        text = "Dueling Noisy DQN Agent with Priority Buffer"
    elif algo == "6":
        text = "DQN n-Steps Agent"
    elif algo == "7":
        text = "DQN Rainbow Agent"
    elif algo == "8":
        text = "Dueling Noisy DQN Agent No PER"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'Algo {text} Number episodes {num_episodes}')
    plt.plot(np.arange(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('Episode #')
    plt.savefig(f'./images/losses_{algo}.jpg')
    return

def plot_scores_training_all():
    """
    plot all scores 2000 episodes
    """
    with open('./outputs/scores.pickle', 'rb') as handle:
        data = pickle.load(handle)
    labels = []
    text = f"DQN Agent ({max(data['1']['scores']).round(2)})"
    labels.append("DQN Agent ")
    num_episodes = "2000"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm scores after {num_episodes} episodes training')
    plt.axhline(y=13, color='r', linestyle='dotted')
    plt.plot(np.arange(len(data['1']['scores'])), data['1']['scores'], label=text)
    text = f"Dueling DQN no Prioritary Buffer Agent ({max(data['2']['scores']).round(2)})"
    labels.append("Dueling DQN no Prioritary Buffer Agent ")
    plt.plot(np.arange(len(data['2']['scores'])), data['2']['scores'], label=text)
    text = f"Dueling DQN with Prioritary Buffer Agent ({max(data['3']['scores']).round(2)})"
    labels.append("Dueling DQN with Prioritary Buffer Agent ")
    plt.plot(np.arange(len(data['3']['scores'])), data['3']['scores'], label=text)
    text = f"Categorical DQN, no Prioritary Buffer Agent ({max(data['4']['scores']).round(2)})"
    labels.append("Categorical DQN, no Prioritary Buffer Agent ")
    plt.plot(np.arange(len(data['4']['scores'])), data['4']['scores'], label=text)
    text = f"Dueling DQN, with Noisy and Prioritary Buffer Agent ({max(data['5']['scores']).round(2)})"
    labels.append("Dueling DQN, with Noisy and Prioritary Buffer Agent ")
    plt.plot(np.arange(len(data['5']['scores'])), data['5']['scores'], label=text)
    text = f"DQN n-Steps Agent ({max(data['6']['scores']).round(2)})"
    labels.append("DQN n-Steps Agent ")
    plt.plot(np.arange(len(data['6']['scores'])), data['6']['scores'], label=text)
    text = f"Rainbow Dueling DQN n-Steps Categorical + Noisy + Prioritary Buffer Agent " \
           f"({max(data['7']['scores']).round(2)})"
    labels.append("Rainbow Dueling DQN n-Steps Categorical + Noisy + Prioritary Buffer Agent ")
    plt.plot(np.arange(len(data['7']['scores'])), data['7']['scores'], label=text)
    text = f"Dueling DQN, with Noisy Layer Agent({max(data['8']['scores']).round(2)})"
    labels.append("Dueling DQN, with Noisy Layer Agent ")
    plt.plot(np.arange(len(data['8']['scores'])), data['8']['scores'], label=text)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    title = "Algorithm and Max Score"
    plt.legend(title=title)
    plt.savefig(f'./images/scores_all.jpg')
    return labels

def plot_losses_training_all():
    """
    plot all losses 2000 episodes
    """
    with open('./outputs/scores.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # add each loss vector and plot it
    text = f"DQN Agent ({min(data['1']['losses']).round(2)})"
    num_episodes = "2000"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm losses after {num_episodes} episodes training')
    # plt.axhline(y=13, color='r', linestyle='dotted')
    plt.plot(np.arange(len(data['1']['losses'])), data['1']['losses'], label=text)
    text = f"Dueling DQN no Prioritary Buffer Agent ({min(data['2']['losses']).round(2)})"
    plt.plot(np.arange(len(data['2']['losses'])), data['2']['losses'], label=text)
    text = f"Dueling DQN with Prioritary Buffer Agent ({min(data['3']['losses']).round(2)})"
    plt.plot(np.arange(len(data['3']['losses'])), data['3']['losses'], label=text)
    text = f"Categorical DQN, no Prioritary Buffer Agent ({min(data['4']['losses']).round(2)})"
    plt.plot(np.arange(len(data['4']['losses'])), data['4']['losses'], label=text)
    text = f"Dueling DQN, with Noisy and Prioritary Buffer Agent ({min(data['5']['losses']).round(2)})"
    plt.plot(np.arange(len(data['5']['losses'])), data['5']['losses'], label=text)
    text = f"DQN n-Steps Agent ({min(data['6']['losses']).round(2)})"
    plt.plot(np.arange(len(data['6']['losses'])), data['6']['losses'], label=text)
    text = f"Rainbow Dueling DQN n-Steps Categorical + Noisy + Prioritary Buffer Agent ({min(data['7']['losses']).round(2)})"
    plt.plot(np.arange(len(data['7']['losses'])), data['7']['losses'], label=text)
    text = f"Dueling DQN, with Noisy Layer Agent ({min(data['8']['losses']).round(2)})"
    plt.plot(np.arange(len(data['8']['losses'])), data['8']['losses'], label=text)
    plt.ylabel('Loss')
    plt.xlabel('Episode #')
    title = "Algorithm and min Loss"
    plt.legend(title=title)

    plt.savefig(f'./images/losses_all.jpg')
    return

def plot_play_scores(labels):
    """

    """

    with open('./outputs/scores.pickle', 'rb') as handle:
        data = pickle.load(handle)

    num_episodes = 2000
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm scores playing with best policy after {num_episodes} episodes training')

    scores = []
    types = []
    for key in data.keys():
        scores.append(data[key]['play'])
        sc = data[key]['play']
        types.append(key)
        plt.bar(int(key), sc, label=labels[int(key) - 1] + str(sc))
    plt.ylabel('Score')
    plt.xlabel('Algorithm #')
    title = "Algorithm and reward in play mode"
    plt.legend(title=title)
    plt.ylim([0, 24])
    plt.tight_layout()

    plt.savefig(f'./images/play_scores_all.jpg')
    return

def plot_time_all(labels):

    """
    plot time to win env . Collect 13 yellow bananas
    """
    with open('./outputs/scores.pickle', 'rb') as handle:
        data = pickle.load(handle)

    num_episodes = 2000
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm time to win the environment. Collect 13 Yellow Banana, during training')

    scores = []
    types = []
    for key in data.keys():
        scores.append(data[key]['time'])
        sc = data[key]['time']
        types.append(key)
        plt.bar(int(key), sc, label=labels[int(key) - 1] + str(round(sc,0)))
    plt.ylabel('Time')
    plt.xlabel('Algorithm #')
    title = "Algorithm and Time to solve Env"
    plt.legend(title=title)
    plt.ylim([0, 20000])
    plt.tight_layout()
    plt.savefig(f'./images/time_scores_all.jpg')

    return

def plot_episodes_to_solve_all(labels):
    """
    plot time to win env . Collect 13 yellow bananas
    """
    with open('./outputs/scores.pickle', 'rb') as handle:
        data = pickle.load(handle)

    num_episodes = 2000
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm number of episodes to win game 13 yellow babanas')

    scores = []
    types = []
    for key in data.keys():
        scores.append(data[key]['time'])
        sc = data[key]['epi']
        types.append(key)
        plt.bar(int(key), sc, label=labels[int(key) - 1] + str(sc))
    plt.ylabel('Num Episodes')
    plt.xlabel('Algorithm #')
    title = "Algorithm and number episodes to solve Env"
    plt.legend(title=title)
    plt.ylim([0, 2000])
    plt.tight_layout()
    plt.savefig(f'./images/episodes_scores_all.jpg')

    return

def save_scores(outcomes, algo, score, episodes, max_t, PER, mode, eps_start, eps_end, eps_decay, fname):
    """
    capture data of a trial and append to outcomes dataframe. Seroalized to csv in folder Monitor

    :param outcomes: Dataframe
    :param algo: number algorithm
    :param score: score (numpy array with scores)
    :param episodes: number of episodes to complete the env
    :param max_t: max steps per episode
    :param PER: Boolean use Prioritary Experience Replay
    :param mode: training or play
    :param eps_start: epsilon start
    :param eps_end: epsilon end
    :param eps_decay: epsilon decay
    :param fname: file name
    :return:
    """
    new = {}
    # cols = ["Algo", "score", "episodes", "max_t", "PER", "mode", "eps_start", "eps_end", "eps_decay"]
    new['Algo'] = algo
    new['score'] = score
    new['episodes'] = episodes
    new['max_t'] = max_t
    new['PER'] = PER
    new['mode'] = mode
    new['eps_start'] = eps_start
    new['eps_end'] = eps_end
    new['eps_decay'] = eps_decay
    # append to output Dataframe
    outcomes = outcomes.append(new, ignore_index=True)
    outcomes.to_csv(fname, index=False)
    return

def load_env(worker_id, base_port, file='./env/Banana.exe'):
    """
    load Unity Environtment
    :param worker_id: ID env
    :param base_port: comunications port with unity agent
    """
    env = UnityEnvironment(file_name=file, worker_id=worker_id, base_port=base_port)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)
    return env , brain_name, brain, action_size, env_info, state, state_size