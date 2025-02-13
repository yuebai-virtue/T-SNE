import argparse
import os

from sklearn import manifold
from sklearn.manifold import TSNE
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def prepro_multi(mat_dis):
    data = scio.loadmat(mat_dis)
    embed = data['r_img']
    label = data['r_l']
    label = label.astype('int')
    result = [Counter(label[i, :]).most_common(1)[0] for i in range(embed.shape[0])]
    ind = []
    for i in range(embed.shape[0]):
        for j in range(2):
            if result[i][j] >= label.shape[1] - 1:
                ind.append(i)
    embed_ = []
    label_ = []
    for i in range(len(ind)):
        embed_.append(embed[ind[i]])
        label_.append(label[ind[i]])

    embed_ = np.array(embed_)
    label_ = np.argmax(np.array(label_), 1)
    l_num = len(list(set(label_)))
    return embed_, label_, l_num


def calc(args, index, choice, noise_list=None):
    point_list = np.empty([args.point_num, len(choice[0])], dtype=int)
    if args.choice:
        thres = args.point_num - args.noise_num
        for j in range(args.point_num):
            if j <= thres:
                point_list[j] = choice[0]
            else:
                point_list[j] = noise_list[j - thres]
    else:
        for j in range(args.point_num):
            if j >= len(choice):
                point_list[j] = choice[0]
            else:
                point_list[j] = choice[j]
    y1 = np.linspace(index, index, args.point_num, dtype=int)
    return point_list, y1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="filename.mat")
    parser.add_argument("--save_dir", type=str, default="./")

    parser.add_argument("--class_num", type=int, default=8, help="classes num for print need")
    parser.add_argument("--choice", type=bool, default=False, help="choice similar embedding point")
    parser.add_argument("--point_num", type=int, default=500, help="point for print need")
    parser.add_argument("--noise_num", type=str, default=20, help="noise point for print need")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    embed, label, lnum = prepro_multi(args.data_dir)

    x, y = [], []
    count = 0
    for i in range(lnum):
        label_list, index_list, embed_list = [], [], []
        for j in range(len(label)):
            if label[j] == i:
                label_list.append(label[j])
                index_list.append(j)
                embed_list.append(embed[j])
        if len(index_list) < args.noise_num:
            continue
        if args.choice:
            embed_choice, embed_noise = [], []
            for j in range(len(embed_list)):
                if (embed_list[j] == embed_list[0]).all():
                    embed_choice.append(embed_list[j])
                else:
                    embed_noise.append(embed_list[j])
            embed_choice = np.array(embed_choice)
            embed_noise = np.array(embed_noise)
            x_, y_ = calc(args, count, embed_choice, embed_noise)
        else:
            x_, y_ = calc(args, count, embed_list)
        x.append(x_)
        y.append(y_)
        count += 1
        if args.class_num != -1 and count >= args.class_num:
            break

    x = np.concatenate(([*x]), axis=0)
    y = np.concatenate(([*y]), axis=0)
    x = torch.from_numpy(x)
    Hamming = calc_hamming_dist(x, x)
    Hamming = Hamming.numpy()
    z = TSNE(n_components=2, perplexity=5, max_iter=1000, learning_rate=10, metric='precomputed',
             init='random').fit_transform(Hamming)
    plt.figure(figsize=(16, 12))
    plt.scatter(z[:, 0], z[:, 1], s=5, c=y)
    os.makedirs(args.save_dir, exist_ok=True)
    file_name = "tsne_" + ("choice" if args.choice else "native") + str(args.point_num) + ".png"
    plt.savefig(os.path.join(args.save_dir, file_name))
    print("finish")
    plt.show()


if __name__ == "__main__":
    main()
