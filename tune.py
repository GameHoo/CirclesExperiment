import os
import torch.nn as nn
import torch
import numpy as np
import ray
from ray import tune
from ray.tune import Trainable
import collections
from circles_experiment.main import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from ray.tune.schedulers import MedianStoppingRule
import pickle
from circles_experiment.network import LSTM, TCN
from circles_experiment.loss import loss1, loss2, loss3, loss4, loss5, loss6

nb_circles = 1000  # 随机生成多少个圆
train_nums = 60000  # 随机多少条数据作为训练集
test_nums = 10000  # 随机抽取多少条数据作为测试集
# 生成随机圆
circles = generate_circles(nums=nb_circles, r_min=10, r_max=100)
# 生成训练集
train_random_circles, train_all_inputs, train_all_targets = generate_data_pairs(circles=circles, l_min=10,
                                                                                l_max=100,
                                                                                nums=train_nums)
# 生成测试集
test_random_circles, test_all_inputs, test_all_targets = generate_data_pairs(circles=circles, l_min=10, l_max=100,
                                                                             nums=test_nums)


class Train(Trainable):
    def _setup(self, config):
        if config['network'] == "LSTM":
            self.model = LSTM(input_size=2, hidden_size=config['hidden_size'], hidden_size2=config['hidden_size2'],
                              device='cuda')
        elif config['network'] == "TCN":
            self.model = TCN(input_size=2, hidden_size=config['hidden_size'], hidden_size2=config['hidden_size2'],
                             device='cuda')
        else:
            raise NotImplementedError
        self.model = self.model.to('cuda')
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=config['lr'])
        self.config = config

    def _train(self):
        global train_nums
        global train_all_inputs, train_all_targets
        global test_random_circles, test_all_inputs, test_all_targets

        batch_size = self.config['batch_size']
        device = 'cuda'
        losses = []
        for i in range(10):
            # 训练集抽取 batch
            batch_idx = torch.randint(0, train_nums, (batch_size,))
            inputs = [train_all_inputs[i] for i in batch_idx]
            targets = [train_all_targets[i] for i in batch_idx]
            # 计算 loss , 反向传播
            self.optim.zero_grad()
            if self.config['loss_func'] == 1:
                loss, _ = loss1(self.model, inputs, targets, device=device, get_predict_target=False)
            elif self.config['loss_func'] == 2:
                loss, _ = loss2(self.model, inputs, targets, device=device, get_predict_target=False)
            elif self.config['loss_func'] == 3:
                loss, _ = loss3(self.model, inputs, targets, device=device, get_predict_target=False)
            elif self.config['loss_func'] == 4:
                loss, _ = loss4(self.model, inputs, targets, device=device, get_predict_target=False)
            elif self.config['loss_func'] == 5:
                loss, _ = loss5(self.model, inputs, targets, device=device, get_predict_target=False)
            elif self.config['loss_func'] == 6:
                loss, _ = loss6(self.model, inputs, targets, device=device, get_predict_target=False)
            else:
                raise NotImplementedError
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        # 计算 测试集 acc
        with torch.no_grad():
            if self.config['loss_func'] == 1:
                _, predict_targets = loss1(self.model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif self.config['loss_func'] == 2:
                _, predict_targets = loss2(self.model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif self.config['loss_func'] == 3:
                _, predict_targets = loss3(self.model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif self.config['loss_func'] == 4:
                _, predict_targets = loss4(self.model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif self.config['loss_func'] == 5:
                _, predict_targets = loss5(self.model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            elif self.config['loss_func'] == 6:
                _, predict_targets = loss6(self.model, test_all_inputs, test_all_targets, device=device,
                                           get_predict_target=True)
            else:
                raise NotImplementedError
            # 计算准确率
            acc = get_acc(test_random_circles, predict_targets, [target[-1] for target in test_all_targets])
        result = {
            'loss_mean': sum(losses) / len(losses),
            'acc': acc
        }
        return result

    def _save(self, tmp_checkpoint_dir):
        model = self.model.to('cpu')
        torch.save(model.state_dict(), os.path.join(tmp_checkpoint_dir, 'model_params.pth'))
        self.model.to('cuda')
        return tmp_checkpoint_dir

    def _restore(self, checkpoint):
        model_params = torch.load(os.path.join(checkpoint, 'model_params.pth'))
        self.model.load_state_dict(model_params)


if __name__ == "__main__":
    ray.init(num_gpus=8, num_cpus=32)

    config = {
        'network': tune.grid_search(['LSTM']),  # 用什么神经网络 (还可以尝试 TCN )
        'loss_func': tune.grid_search(([1, 2, 3, 4, 5, 6])),  # 用什么 loss 函数
        'batch_size': 100,  # 每次抽取多少条训练
        'lr': tune.loguniform(1e-6, 1e-1),
        'hidden_size': tune.grid_search([128]),  # 神经网络 隐藏层单元数
        'hidden_size2': tune.grid_search([16, 64])
    }
    tune.run(Train, config=config, resources_per_trial={"cpu": 1, "gpu": 0.3},
             local_dir=os.path.abspath(os.path.join(__file__, os.pardir, 'TRY_ALL')), verbose=1,
             stop={
                 "training_iteration": 2000
             },
             num_samples=10,
             checkpoint_freq=200,
             checkpoint_at_end=True,
             scheduler=MedianStoppingRule(time_attr='training_iteration', metric='acc', mode='max',
                                          grace_period=200)
             )
