"""
    主要流程：
    1. 生成 指定个数，随机半径的同心圆。
    2. 从同心圆抽取 训练集 （X，Y）
        X ：在随机某个圆取一个随机起点，在圆上以逆时针间隔1度截取随机长度的坐标序列。
        Y ：下一个点的坐标。
        （ 可以在 X 中加入噪声 ）
    3. 用神经网络训练
    4. 从同心圆抽取测试集，判断准确度。
        准确度判断方法：
            以目标点为圆心，间隔1度的点的直线为直径的圆内。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from circles_experiment.network import LSTM, TCN
from circles_experiment.data_process import generate_circles, generate_data_pairs, get_rise, get_acc
from circles_experiment.loss import loss1, loss2, loss3, loss4, loss5, loss6


def test_train():
    # 超参数
    device = 'cuda:0'
    nb_circles = 1000  # 随机生成多少个圆
    nb_epoches = 10000000  # 训练多少个epoch
    nb_train_steps = 100  # 每个epoch 训练多少次
    train_nums = 60000  # 随机多少条数据作为训练集
    batch_size = 1000  # 每次抽取多少条训练
    test_nums = 10000  # 随机抽取多少条数据作为测试集
    lr = 5e-4
    loss_func = 1
    # 生成随机圆
    circles = generate_circles(nums=nb_circles, r_min=10, r_max=100)
    # 生成训练集
    train_random_circles, train_all_inputs, train_all_targets = generate_data_pairs(circles=circles, l_min=10,
                                                                                    l_max=100,
                                                                                    nums=train_nums)
    # 生成测试集
    test_random_circles, test_all_inputs, test_all_targets = generate_data_pairs(circles=circles, l_min=10, l_max=100,
                                                                                 nums=test_nums)

    # 初始化神经网络
    model = LSTM(input_size=2, hidden_size=128, hidden_size2=16, device=device)
    model.to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    for t_epoch in range(nb_epoches):
        for t_train_step in range(nb_train_steps):
            # 训练集抽取 batch
            batch_idx = torch.randint(0, train_nums, (batch_size,))
            inputs = [train_all_inputs[i] for i in batch_idx]
            targets = [train_all_targets[i] for i in batch_idx]
            # 计算 loss , 反向传播
            optim.zero_grad()
            if loss_func == 1:
                loss, _ = loss1(model, inputs, targets, device=device, get_predict_target=False)
            elif loss_func == 2:
                loss, _ = loss2(model, inputs, targets, device=device, get_predict_target=False)
            elif loss_func == 3:
                loss, _ = loss3(model, inputs, targets, device=device, get_predict_target=False)
            elif loss_func == 4:
                loss, _ = loss4(model, inputs, targets, device=device, get_predict_target=False)
            elif loss_func == 5:
                loss, _ = loss5(model, inputs, targets, device=device, get_predict_target=False)
            elif loss_func == 6:
                loss, _ = loss6(model, inputs, targets, device=device, get_predict_target=False)
            else:
                raise NotImplementedError
            loss.backward()
            optim.step()
            print("loss: %f" % loss.item())

        # 计算 测试集 acc
        with torch.no_grad():
            if loss_func == 1:
                _, predict_targets = loss1(model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif loss_func == 2:
                _, predict_targets = loss2(model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif loss_func == 3:
                _, predict_targets = loss3(model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif loss_func == 4:
                _, predict_targets = loss4(model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif loss_func == 5:
                _, predict_targets = loss5(model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif loss_func == 6:
                _, predict_targets = loss6(model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            else:
                raise NotImplementedError
            # 计算准确率
            acc = get_acc(test_random_circles, predict_targets, [target[-1] for target in test_all_targets])
            print("acc: %f" % acc)


if __name__ == "__main__":
    test_train()
