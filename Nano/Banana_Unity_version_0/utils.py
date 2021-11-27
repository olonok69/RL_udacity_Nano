import numpy as np
import matplotlib.pyplot as plt

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
        text = "Double DQN Agent"
    elif algo == "4":
        text = "Dueling DQN Agent"
    elif algo == "2":
        text = "Dueling DQN Agent with Priority Buffer"
    elif algo == "5":
        text = "Dueling Noisy DQN Agent with Priority Buffer"
    elif algo == "6":
        text = "Categorical DQN Agent"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'Algo {text} Number episodes {num_episodes}')
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'./images/scores_{algo}.jpg')
    return

def save_scores(outcomes, algo, score, episodes, max_t, PER, mode, eps_start, eps_end, eps_decay, fname):
    """
    capture data of a trial and append to outcomes dataframe. Seroalized to csv in folder Monitor
    :param outcomes:
    :param algo:
    :param score:
    :param episodes:
    :param max_t:
    :param PER:
    :param mode:
    :param eps_start:
    :param eps_end:
    :param eps_decay:
    :param fname:
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