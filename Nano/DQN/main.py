from utils import *
import gym
import tensorflow as tf
# gpu_options = tf.GPUOptions(allow_growth=True)  # init TF ...
# config=tf.ConfigProto(gpu_options=gpu_options)  # w/o taking ...
# with tf.Session(config=config): pass

def experiment_mountaincar():
    neural_net = TFNeuralNet(nb_in=2, nb_hid_1=64, nb_hid_2=64, nb_out=3, lr=0.00025)

    model = TFFunctApprox(neural_net,
                          env.observation_space.low,
                          env.observation_space.high,
                          rew_mean=-50,
                          rew_std=15,
                          nb_actions=env.action_space.n)

    mem = Memory(max_len=100000, state_shape=(2,), state_dtype=float)
    mem_fill(env, mem, steps=10000)
    test_states, _, _, _, _, _ = mem.get_batch(10)

    trace = Trace(eval_every=1000, test_states=test_states)

    return trace, model, mem

env = gym.make('MountainCar-v0').env  # remove 200 step limit
env = WrapFrameSkip(env, frameskip=4)
trace, model, mem = experiment_mountaincar()
trace.enable_plotting = True

tf.compat.v1.disable_eager_execution()

tts = q_learning(env, frames=25000, gamma=.99,
                 eps_decay_steps=20000, eps_target=0.1, batch_size=4096,
                 model=model, mem=mem, callback=callback, trace=trace)

model._model.save('./tf_models/mc/MountainCar.ckpt')

model._model.load('./tf_models/mc/MountainCar.ckpt')


try: evaluate(env, model, frames=float('inf'), eps=0.0, render=True)
except KeyboardInterrupt: pass
finally: env.close()




