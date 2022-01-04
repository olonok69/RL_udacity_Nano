import gym
from agents import A2CAgent, ActionNormalizer, PPOAgent, DDPGAgent, TD3Agent, SACAgent
import argparse
import torch


def main():
    # Command line Arguments
    parser = argparse.ArgumentParser("TDL")
    parser.add_argument("--mode", type=str, help="training , play, compare, complare_play, plot, hp_tuning",
                        required=True)
    parser.add_argument("--type", type=str, help="type 1-->A2C , type 2--> PPO, type 3--> DDPG"
                                                 "type 4-->TD3, type 5--> Duelling DQN"
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
    elif args.mode == "training" and algo == "3":
        env = ActionNormalizer(env)
        env.seed(seed)
        # parameters
        num_frames = 50000
        memory_size = 100000
        batch_size = 128
        ou_noise_theta = 1.0
        ou_noise_sigma = 0.1
        initial_random_steps = 10000
        agent = DDPGAgent(
            env,
            memory_size,
            batch_size,
            ou_noise_theta,
            ou_noise_sigma,
            initial_random_steps=initial_random_steps
        )
        actor_model, critic_model = agent.train(num_frames)
        torch.save(actor_model.state_dict(), f'models\checkpoint_actor_{algo}.pth')
        torch.save(critic_model.state_dict(), f'models\checkpoint_critic_{algo}.pth')
    elif args.mode == "training" and algo == "4":
        env = ActionNormalizer(env)
        env.seed(seed)
        # parameters
        num_frames = 50000
        memory_size = 100000
        batch_size = 128
        initial_random_steps = 10000

        agent = TD3Agent(
            env, memory_size, batch_size, initial_random_steps=initial_random_steps
        )
        actor_model = agent.train(num_frames)
        torch.save(actor_model.state_dict(), f'models\checkpoint_actor_{algo}.pth')
    elif args.mode == "training" and algo == "5":
        env = ActionNormalizer(env)
        env.seed(seed)
        # parameters
        num_frames = 50000
        memory_size = 100000
        batch_size = 128
        initial_random_steps = 10000

        agent = SACAgent(
            env, memory_size, batch_size, initial_random_steps=initial_random_steps
        )
        actor_model = agent.train(num_frames)
        torch.save(actor_model.state_dict(), f'models\checkpoint_actor_{algo}.pth')

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
    elif args.mode == "play" and algo == "3":
        env = ActionNormalizer(env)
        env.seed(seed)
        # parameters
        num_frames = 50000
        memory_size = 100000
        batch_size = 128
        ou_noise_theta = 1.0
        ou_noise_sigma = 0.1
        initial_random_steps = 10000
        agent = DDPGAgent(
            env,
            memory_size,
            batch_size,
            ou_noise_theta,
            ou_noise_sigma,
            initial_random_steps=initial_random_steps
        )
        agent.actor.load_state_dict(torch.load(f'models\checkpoint_actor_{algo}.pth'))
        agent.env = gym.wrappers.RecordVideo(agent.env, "videos\ddpg")

        frames, scores = agent.test()
        print(scores)
    elif args.mode == "play" and algo == "4":
        env = ActionNormalizer(env)
        env.seed(seed)
        # parameters
        num_frames = 50000
        memory_size = 100000
        batch_size = 128
        initial_random_steps = 10000

        agent = TD3Agent(
            env, memory_size, batch_size, initial_random_steps=initial_random_steps
        )
        agent.actor.load_state_dict(torch.load(f'models\checkpoint_actor_{algo}.pth'))
        agent.env = gym.wrappers.RecordVideo(agent.env, "videos\\td3")

        frames, scores = agent.test()
        print(scores)
    elif args.mode == "play" and algo == "5":
        env = ActionNormalizer(env)
        env.seed(seed)
        # parameters
        num_frames = 50000
        memory_size = 100000
        batch_size = 128
        initial_random_steps = 10000

        agent = SACAgent(
            env, memory_size, batch_size, initial_random_steps=initial_random_steps
        )
        agent.actor.load_state_dict(torch.load(f'models\checkpoint_actor_{algo}.pth'))
        agent.env = gym.wrappers.RecordVideo(agent.env, "videos\\sac")

        frames, scores = agent.test()
        print(scores)
if __name__ == '__main__':
    main()