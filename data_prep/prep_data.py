import scipy.io as sio
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from collections import Counter
import math
# import statistics as stats
# import functions
import os
import csv
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=False, help='cifar10 | mnist | fmnist '| svhn', default='mnist')
parser.add_argument('--f', help='path to folder', default='', required=True)
parser.add_argument('--y', type=int, help='value of y', required=True)
parser.add_argument('--random', type=bool, help='augment by random', required=False, default=False)
parser.add_argument('--stride', type=int, help='stride of augmentation', required=False, default=0)
parser.add_argument('--noise', type=bool, help='add a noise class', required=False, default=False)
opt = parser.parse_args()
print(opt)

folder = opt.f #'data_raw/non-person'
id = 0
for filename in os.listdir(folder):
    id+=1
    fl = folder+'./'+filename
    b = pd.read_csv(fl, header=None)

    dp_c = []
    snr_c = []
    # t = []
    dp = b.iloc[1,:]
    snr = b.iloc[3,:]
    time = b.iloc[0,:]
    for i in b.iloc[4,:]:
        dp_c.append(np.mean(dp[time==i]))
        snr_c.append(np.mean(snr[time==i]))
    dp_c = [x for x in dp_c if str(x) != 'nan']
    snr_c = [x for x in snr_c if str(x) != 'nan']

    angle_c = b.iloc[8,:]
    num_pc = b.iloc[7,:]
    norm_range_std = b.iloc[6,:]
    angle_c = [x for x in angle_c if str(x) != 'nan']
    num_pc = [x for x in num_pc if str(x) != 'nan']
    norm_range_std = [x for x in norm_range_std if str(x) != 'nan']

    try:
        data = np.column_stack((dp_c,snr_c, angle_c, num_pc, norm_range_std))

        # y = np.ones((data.shape[0])) * opt.y  # CHECK

        # data = np.column_stack((data, y))

        for i in range(0,len(data)-16):
            data = np.row_stack((data, data[i:i+15,:]))

        if id == 1:
            print("yeah")
            final_data = data
        else:
            final_data = np.row_stack((final_data, data))
    except Exception as e:
        print(filename, len(dp_c), len(snr_c), len(angle_c), len(num_pc), len(norm_range_std), e)

if opt.random:
    # dp_r = np.random.random_sample((final_data.shape[0]*6)) * np.max(final_data[:,0])
    # snr_r = np.random.random_sample((final_data.shape[0]*6)) * np.max(final_data[:,1])
    # angle_r = np.random.random_sample((final_data.shape[0]*6)) * np.max(final_data[:,2])
    # num_r = np.random.random_sample((final_data.shape[0]*6)) * np.max(final_data[:,3])
    # norm_r = np.random.random_sample((final_data.shape[0]*6)) * np.max(final_data[:,4])

    dp_r = np.random.uniform(low=np.min(final_data[:,0]), high=np.max(final_data[:,0]), size = (final_data.shape[0]*6))
    snr_r = np.random.uniform(low=np.min(final_data[:,1]), high=np.max(final_data[:,1]), size = (final_data.shape[0]*6))
    angle_r = np.random.uniform(low=np.min(final_data[:,2]), high=np.max(final_data[:,2]), size = (final_data.shape[0]*6))
    num_r = np.random.uniform(low=np.min(final_data[:,3]), high=np.max(final_data[:,3]), size = (final_data.shape[0]*6))
    norm_r = np.random.uniform(low=np.min(final_data[:,4]), high=np.max(final_data[:,4]), size = (final_data.shape[0]*6))

    R = np.column_stack((dp_r, snr_r, angle_r, num_r, norm_r))

    # y = np.ones((R.shape[0])) * opt.y

    # R = np.column_stack((R, y))

    final_data = np.row_stack((final_data, R))

    # for i in range(len(final_data)):
    #     final_data = np.row_stack((final_data, final_data[i:i+7,:]))
    #     if i+8 == len(final_data):
    #         break
    # i+=1
train_x = []
train_y = []

for i in range(final_data.shape[0]-15):
    train_x.append(final_data[i:i+16,:])
    train_y.append(np.ones((1))*opt.y)

if opt.noise:
    dp_r = np.random.uniform(low=np.min(final_data[:,0]), high=np.max(final_data[:,0]), size = (final_data.shape[0]))
    snr_r = np.random.uniform(low=np.min(final_data[:,1]), high=np.max(final_data[:,1]), size = (final_data.shape[0]))
    angle_r = np.random.uniform(low=np.min(final_data[:,2]), high=np.max(final_data[:,2]), size = (final_data.shape[0]))
    num_r = np.random.uniform(low=np.min(final_data[:,3]), high=np.max(final_data[:,3]), size = (final_data.shape[0]))
    norm_r = np.random.uniform(low=np.min(final_data[:,4]), high=np.max(final_data[:,4]), size = (final_data.shape[0]))

    R = np.column_stack((dp_r, snr_r, angle_r, num_r, norm_r))

    for i in range(final_data.shape[0]-15):
        train_x.append(R[i:i+16,:])
        train_y.append(np.ones((1))*5)

print(final_data.shape)
print(len(train_x), train_x[-1].shape, train_x[0].shape, len(train_y))

fil = opt.f[9:]+'.pkl'
with open(fil, "wb") as f:
    pickle.dump((train_x,train_y), f)
# with open('crouching_test.pkl', "wb") as f:   # for tests
#     pickle.dump((train_x,train_y), f)