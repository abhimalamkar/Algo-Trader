#!/usr/bin/env python3
import argparse
import numpy as np

from environ import StocksEnv, Actions, TRADIN_INTERVAL, PAGE
from ga_batch_ import Net, dataDir, FILE_NAME
from datetime import datetime
import copy
import pandas_datareader.data as web
import os
import torch

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import pickle

company = 'F'
start = datetime(1900, 1, 1)
end = datetime.now()

EPSILON = 0.02
NOISE_STD = 0.005

def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net

def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape[0], env.action_space.n)
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net

def load(filename):
    infile = open(filename,'rb')
    seeds = pickle.load(infile)
    infile.close()
    return seeds

def save(filename,file):
    outfile = open(filename,'wb')
    pickle.dump(file,outfile)
    outfile.close()

def buildDir(directory):
    try:
        if (not os.path.isdir(directory)):
            os.mkdir(directory)
    except OSError:
        print ("Creation of the directory %s failed" % directory)
    else:
        print ("Successfully created the directory %s " % directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data", required=True, help="CSV file with quotes to run the model")
    #parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-r", "--rev")
    parser.add_argument("-n", "--name", required=True, help="Name to use in output images")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission size in percent, default=0.1")
    args = parser.parse_args()

    company = args.name

    buildDir("../outputs/" + company)

    df = web.DataReader(company,'yahoo',start=start,end=end)
    df.to_csv("../data/"+company+".csv")
    #print(df.shape,df.head())
    #df = pd.read_csv('EURUSD_M1_201911.csv' ,nrows=4000,names=['Time','minute','Close','high',"low","open","s"])
    # df = pd.read_csv("../data/PLUG.csv",)
    
    if (args.rev):
        print("REversed")
        df = df.iloc[::-1]
    close = df.Close.values.tolist()
    elite = load('./'+dataDir+'/population.data')[0]
    env = StocksEnv(close,test=True)
    net = build_net(env, elite[0])#Net(env.observation_space.shape[0], env.action_space.n)
    #net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net.zero_noise(batch_size=1)
    obs = env.reset()
    
    #df_ = df.iloc[env._offset:env._offset + TRADIN_INTERVAL]
    start_price = env._cur_close()

    total_reward = 0.0
    step_idx = 0
    total_rewards = []
    rewards = []

    while True:
        step_idx += 1
        obs_v = torch.FloatTensor([obs])
        out_v = net(obs_v)
        action_idx = np.argmax(out_v.data.numpy()[0])

        action = Actions(action_idx)
        
        obs, reward, done, info = env.step(action_idx)
        #print(step_idx,action,reward)
        total_reward += reward
        rewards.append(reward)
        total_rewards.append(total_reward)

        if step_idx % 100 == 0:
            print("%d: reward=%.3f, total reward=%.3f" % (step_idx, reward, total_reward))
        if done:
            print("%d: reward=%.3f, total reward=%.3f" % (step_idx, reward, total_reward))
            break
    # df = pd.read_csv(args.data)
    
    #fig = plt.figure(figsize = (150, 50))
    
    # plt.plot(df['Close'], '^', markersize=2, color='g', label = 'buying signal', markevery = states_buy)
    # plt.plot(df['Close'], 'v', markersize=2, color='r', label = 'selling signal', markevery = states_sell)

    # plt.plot(df['Close'], label = 'true close', c = 'g')
    # plt.plot(df['Close'], 'X', label = 'predict buy', markevery = states_buy, c = 'b')
    # plt.plot(df['Close'], 'o', label = 'predict sell', markevery = states_sell, c = 'r')
    plt.clf()
    plt.plot(close)
    plt.savefig("../outputs/"+company+"/graph-%s.png" % args.name)

    # plt.clf()
    # plt.plot(df['Close'])
    # plt.savefig("sprofit-%s.png" % args.name)
    plt.clf()
    plt.plot(rewards)
    plt.savefig("../outputs/"+company+"/rewards_values-%s.png" % args.name)

    plt.clf()
    plt.plot(total_rewards)
    plt.title("Total reward, data=%s" % args.name)
    plt.ylabel("Reward, %")
    plt.savefig("../outputs/"+company+"/rewards-%s.png" % args.name)
