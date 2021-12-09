import gym
from agents import A2CAgent, ActionNormalizer, PPOAgent
import argparse
import torch


def main():
    # Command line Arguments
    parser = argparse.ArgumentParser("DQN")
    parser.add_argument("--mode", type=str, help="training , play, compare, complare_play, plot, hp_tuning",
                        required=True)
    parser.add_argument("--type", type=str, help="type 1-->A2C , type 2--> PPO, type 3--> Dueling"
                                                 "DQN, no PBR, type 4-->categorical DQN, type 5--> Duelling DQN"
                                                 " with Noisy layer and PBR, Type 6--> DQN n-steps, type 7 --> "
                                                 "Rainbow DQN", required=True)
    args = parser.parse_args()

    seed = 777
    # load environment
    if args.mode != "compare" and args.mode != "compare_play" and args.mode != "plot" and args.mode != "hp_tuning":

        env_id = "Pendulum-v0"
        env = gym.make(env_id)

    algo = args.type# <-- type of algo to use for training
    if  args.mode =="training" and algo =="1":
        num_frames = 200000
        gamma = 0.99
        entropy_weight = 1e-2

        agent = A2CAgent(env, gamma, entropy_weight)

        agent.train(num_frames)
    elif args.mode =="training" and algo == "2":
        env = ActionNormalizer(env)
        env.seed(seed)
        # parameters
        num_frames = 100000

        agent = PPOAgent(
            env,
            gamma=0.9,
            tau=0.8,
            batch_size=64,
            epsilon=0.2,
            epoch=64,
            rollout_len=2048,
            entropy_weight=0.005
        )
        actor_model, critic_model = agent.train(num_frames)
        torch.save(actor_model.state_dict(), f'models\checkpoint_actor_{algo}.pth')
        torch.save(critic_model.state_dict(), f'models\checkpoint_critic_{algo}.pth')
    elif args.mode =="play" and algo =="2":
        env = ActionNormalizer(env)
        env.seed(seed)
        agent = PPOAgent(
            env,
            gamma=0.9,
            tau=0.8,
            batch_size=64,
            epsilon=0.2,
            epoch=64,
            rollout_len=2048,
            entropy_weight=0.005
        )
        agent.actor.load_state_dict(torch.load(f'models\checkpoint_actor_{algo}.pth'))
        agent.env = gym.wrappers.RecordVideo(agent.env, "videos\ppo")

        frames, scores = agent.test()
        print(scores)

if __name__ == '__main__':
    main()