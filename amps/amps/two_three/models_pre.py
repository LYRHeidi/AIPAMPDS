import datetime
import os

from func.two_three.model import ETFC
from model import *

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch


def ArgsGet(fa_path):
    parse = argparse.ArgumentParser()
    # parse.add_argument('-file', type=str, default='/home/zwq/combine_models_copy/Ec_Sa_mic_model/example/example.fasta', help='fasta file')
    # parse.add_argument('-out_path', type=str, default='/home/zwq/combine_models_copy/Ec_Sa_mic_model/example',  help='output path')
    parse.add_argument('-file', type=str, default=fa_path, help='fasta file')
    parse.add_argument('-out_path', type=str, default='./example', help='output path')
    args = parse.parse_args()
    return args


def get_data(file):
    seqs = []
    names = []
    seq_length = []
    with open(file) as f:
        for each in f:
            if each == '\n':
                continue
            elif each[0] == '>':
                names.append(each.strip())
            else:
                seqs.append(each.rstrip())

    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    max_len = 50
    data_e = []
    to_delete = []  # 用于存储需要删除的索引

    for i in range(len(seqs)):
        if len(seqs[i]) > max_len or len(seqs[i]) < 5:
            print('本方法只能识别序列长度在5-50AA的多肽,该序列将不能识别:{}'.format(seqs[i]))
            to_delete.append(i)
            continue

        length = len(seqs[i])
        seq_length.append(length)
        elemt = []
        for j in seqs[i]:
            if j == ',' or j == '1' or j == '0':
                continue
            elif j not in amino_acids:
                print('本方法只能识别包含天然氨基酸的多肽,该序列不能识别:{}'.format(seqs[i]))
                to_delete.append(i)
                break
            else:
                index = amino_acids.index(j)
                elemt.append(index)
        else:
            if length <= max_len:
                elemt += [0] * (max_len - length)
                data_e.append(elemt)

    # 删除标记的元素
    for i in sorted(to_delete, reverse=True):
        del names[i]
        del seqs[i]

    return np.array(data_e), seqs, np.array(seq_length), names


def predict(test, seq_length, device, model_dir):
    model = ETFC(50, 192, 0.4, 1, 8)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device)
    test = test.to(device)
    seq_length = seq_length.to(device)

    model.eval()
    with torch.no_grad():
        score_label = model(test, seq_length)

    return score_label.cpu().numpy()  # 返回概率值，不返回二分类标签


def pre_my(test_data, seq_length, output_path, seqs, device):
    batch_size = 1000
    scores_list_1 = []
    scores_list_2 = []

    # 第一个模型 EC模型
    ec_model_dir = './model/ec_model.h5'
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]
        batch_seq_length = seq_length[i:i + batch_size]
        scores = predict(batch_data, batch_seq_length, device, ec_model_dir)
        # scores_list_2.extend(scores)
        scores_list_1.extend([float(score) for score in scores])

    # 第二个模型 SA模型
    sa_model_dir = './model/sa_model.h5'  # 替换为第一个模型的路径
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]
        batch_seq_length = seq_length[i:i + batch_size]
        scores = predict(batch_data, batch_seq_length, device, sa_model_dir)
        # scores_list_1.extend(scores)
        scores_list_2.extend([float(score) for score in scores])

    amp_seqs = []
    for i in range(len(scores_list_1)):  #
        # if scores_list_1[i][0] >= 0.9980798959732056 and scores_list_2[i][0] >= 0.9989857077598572:  # 这里设置两个模型的阈值
        if scores_list_1[i] >= 0.9980798959732056 and scores_list_2[i] >= 0.9989857077598572:
            amp_seqs.append(seqs[i])

    # 生成 TXT 文件
    output_file = os.path.join(output_path, 'AMP_sequences.txt')
    with open(output_file, 'w') as f:
        for seq in amp_seqs:
            f.write(seq + '\n')

    # 生成 CSV 文件
    csv_output_file = os.path.join(output_path, 'predictions.csv')
    obj = {
        'optimal': [],
        'all': [],
    }

    for i in amp_seqs:
        obj['optimal'].append(i)

    for k, i in enumerate(seqs):
        obj['all'].append({
            'Sequence': i,
            'E.coli_model_score': '{:.4f}'.format(scores_list_1[k]),
            'S.aureus_model_score': '{:.4f}'.format(scores_list_2[k]),
        })
    data = {
        'Sequence': seqs,
        'E.coli_model_score': scores_list_1,
        'S.aureus_model_score': scores_list_2
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_output_file, index=False)
    return obj


# if __name__ == '__main__':
def for_t_t(fa_path):
    start = datetime.datetime.now()
    print('-----------------开始时间-----------------------')
    print(start)

    args = ArgsGet(fa_path)
    print('args')
    file = args.file  # fasta file
    print('file')
    output_path = args.out_path  # output path
    print('output_path')
    Path(output_path).mkdir(exist_ok=True)
    print(output_path)
    data, seqs, seq_length, names = get_data(file)
    print('get_data')
    data = torch.LongTensor(data)
    print('torch.LongTensor')
    seq_length = torch.LongTensor(seq_length)
    print('seq_length')
    print(datetime.datetime.now())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device(cuda:1 if torch.cuda.is_available() else 'cpu')
    data = pre_my(data, seq_length, output_path, seqs, device)
    print(datetime.datetime.now())
    print('pre_my')
    end = datetime.datetime.now()
    print('-----------------结束时间-----------------------')
    print(end)
    runTime = end - start
    print('-----------------运行时间-----------------------')
    print(runTime)
    return data
