#!/usr/bin/env python3
import sys
import gym
import roboschool
import collections
import copy
import time
import numpy as np
import pandas as pd
import pickle
import torch
import random
import torch.nn as nn
import torch.multiprocessing as mp

from datetime import datetime

from tensorboardX import SummaryWriter
from environ import StocksEnv, Actions

NOISE_STD = 0.1
POPULATION_SIZE = 20
PARENTS_COUNT = 10
WORKERS_COUNT = 8
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1
EVIROMENT_FILE = "../data/coinbase_hourly.csv"

class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(obs_size, 512),
            nn.Linear(512, 1024),
            # nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, action_size),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        action_v = net(obs_v)
        obs, r, done, _ = env.step(np.argmax(action_v.data.numpy()[0]))
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net

def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape[0], env.action_space.n)
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'reward', 'steps'])

def build_env(filename):
    df = pd.read_csv(filename)
    #df = pd.read_csv('EURUSD_M1_201911.csv',nrows=4000 ,names=['Time','minute','Close','high',"low","open","s"])
    # randomly change direction
    # if random.choice([True, False]):
    #     df = df.iloc[::-1]

    close = df.Close.values.tolist()
    return StocksEnv(close)

def save(filename,file):
    outfile = open(filename,'wb')
    pickle.dump(file,outfile)
    outfile.close()

def load(filename):
    infile = open(filename,'rb')
    seeds = pickle.load(infile)
    infile.close()
    return seeds

def worker_func(input_queue, output_queue):
    env = build_env(EVIROMENT_FILE)
    cache = {}

    while True:
        parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1])
                else:
                    net = build_net(env, net_seeds)
            else:
                net = build_net(env, net_seeds)
            new_cache[net_seeds] = net
            reward, steps = evaluate(env, net)
            output_queue.put(OutputItem(seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache

def test(env,population,writer):
    rewards = []

    for i in range(PARENTS_COUNT):
        agent = population[i]
        net = build_net(env,agent[0])

        #env._set_seed(-1)
        obs = env.reset()
        env._test_env = True
        
        total_reward = 0.0
        step_idx = 0

        states_buy = []
        states_sell = []

        total_trades = 0
        
        while True:
            step_idx += 1
            obs_v = torch.FloatTensor([obs])
            out_v = net(obs_v)
            action_idx = np.argmax(out_v.data.numpy()[0])

            action = Actions(action_idx)

            obs, reward, done, info = env.step(action_idx)

            if action.Buy:
                states_buy.append(info["offset"])
                total_trades += 1

            if action.Close:
                states_sell.append(info["offset"])

            total_reward += reward
            
            if done:
                epoch_time = int(time.time())
                rewards.append(total_reward)
                writer.add_scalar("reward parent %d"%i, total_reward, epoch_time)
                print("Reward: = %.2f" % (total_reward))
                break

    writer.add_scalar("rewards mean", np.mean(rewards), epoch_time)
    print("Mean reward: %.2f" % np.mean(rewards))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment=EVIROMENT_FILE)

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None#load('./models/MP/elite.data')
    load_population = False

    while True:
        t_start = time.time()
        batch_steps = 0

        population = []

        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps

        if elite is not None:
            population.append(elite)

        if load_population:
            population += load('./models/MP copy/population.data')
            load_population = False

        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        # env = build_env(EVIROMENT_FILE)
        # net = build_net(env,elite[0])
        # torch.save(net.state_dict(), './models/EUR/%d_elite_net.model'%elite[1])

        # save('./models/EUR/elite.data',elite)
        # save('./models/EUR/population.data',population)

        # if gen_idx%10 == 0:
        #     test(env,population,writer)

        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
            worker_queue.put(seeds)
        gen_idx += 1

    pass
