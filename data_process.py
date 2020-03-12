from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np
import math
import matplotlib.pyplot as plt


class Point(object):
    def __init__(self, x, y):
        """
        :param x: 直角坐标系x,坐标
        :param y: 直角坐标系y,坐标
        """
        self.x = x
        self.y = y


def polar_to_rectangular(r, theta):  # 极坐标转化为直角坐标
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y


def rectangular_to_polar(x, y):  # 直角坐标转化为极坐标
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(y, x) + math.pi  # 返回 (0,2*pi] 的弧度
    return r, theta


def angle_to_radian(angle):
    """
    角度转弧度
    :return:
    """
    return angle / 180 * math.pi


class Circle(object):
    # 一个圆
    def __init__(self, radius, center: Point):
        """
        :param radius: 半径
        :param center: 圆心（直角坐标系）
        """
        self.radius = radius
        self.center = center


def generate_circles(nums=4000, r_min=10, r_max=1000, x_min=-100, x_max=100, y_min=-100, y_max=100):
    """
    生成 指定个数，随机半径的同心圆。
    :param nums: 生成的圆的个数
    :param r_min: 最小半径
    :param r_max: 最大半径
    :param x_min: 圆心坐标范围
    :param x_max: 圆心坐标范围
    :param y_min: 圆心坐标范围
    :param y_max: 圆心坐标范围
    :return: list, 圆的半径列表
    """
    circles = []
    for i in range(nums):
        radius = np.random.random() * (r_max - r_min) + r_min
        x = np.random.random() * (x_max - x_min) + x_min
        y = np.random.random() * (y_max - y_min) + y_min
        center = Point(x, y)
        circles.append(Circle(radius, center))
    return circles


def generate_data_pairs(circles, l_min=5, l_max=20, nums=1000, angle_interval=1):
    """
    在指定的 circles 里，随机抽数据集。（用来训练或者测试）
    :param circles: list, 圆的半径列表
    :param l_min: 序列的最小长度
    :param l_max: 序列的最大长度
    :param nums: 生成多少个数据对
    :param angle_interval: 序列角度每次增加多少度，默认 1 度。
    :return:
        random_circles: 生成数据使用的圆
        all_inputs: 一个列表，包含nums个输入数据，每个输入数据的形式为：[[x1,y1],[x2,y2] ........]
        all_targets: 一个列表，包含nums个标签数据，每个标签数据的形式为：[[x1,y1],[x2,y2] ........]
    """
    result = []
    random_starts = np.random.randint(0, 360, size=nums)  # 随机起点
    random_circles = np.random.choice(circles, size=nums, replace=True)  # 随机圆
    random_lens = np.random.randint(l_min, l_max + 1, size=nums)  # 随机输入序列长度, 范围 [l_min, l_max]
    all_inputs = []  # 多个输入序列的列表
    all_targets = []  # 多个标签的列表
    for i in range(nums):
        one_input = []  # 第i组数据的 输入序列
        one_target = []  # 第i组数据的 标签
        for j in range(random_lens[i]):
            x, y = polar_to_rectangular(r=random_circles[i].radius,
                                        theta=angle_to_radian(random_starts[i] + j * angle_interval))
            x += random_circles[i].center.x
            y += random_circles[i].center.y
            one_input.append([x, y])
            # 后面一个点的坐标当作当前输出
            x, y = polar_to_rectangular(r=random_circles[i].radius,
                                        theta=angle_to_radian(random_starts[i] + (j + 1) * angle_interval))
            x += random_circles[i].center.x
            y += random_circles[i].center.y
            one_target.append([x, y])
        all_inputs.append(one_input)
        all_targets.append(one_target)
    return random_circles, all_inputs, all_targets


def plot_data_pairs(all_inputs, all_targets):
    """
    把 generate_data_pairs 生成的数据绘制出来 （调试用）
    """
    plt.figure()
    for i in range(len(all_inputs)):
        one_input = all_inputs[i]
        one_target = all_targets[i]
        plt.plot([point[0] for point in one_input], [point[1] for point in one_input], label='input_%d' % i)
        # plt.plot([point[0] for point in one_target], [point[1] for point in one_target], label='target_%d' % i)
        # 绘制最后一个 target
        last_target = all_targets[i][-1]
        plt.plot(last_target[0], last_target[1], 'o', label='output_%d' % i)
    plt.legend()
    plt.show()


def test_plot_data_pairs():
    """
    生成一些数据对 并绘制出来
    :return:
    """
    circles = generate_circles(nums=100, r_min=10, r_max=100)
    _, all_inputs, all_targets = generate_data_pairs(circles=circles, l_min=150, l_max=300, nums=5, angle_interval=1)
    plot_data_pairs(all_inputs, all_targets)


def get_rise(sequences):
    """
    把 generate_data_pairs 生成的 all_inputs 或者 all_targets 转化为波动率
    :return:
        first_items : list, sequences 里 每个 sequence 的第一个数据 的列表（用于数据还原）
        new_sequences : list, 转化为波动率之后的 sequences
    """
    eps = 1e-11
    first_items = []
    new_sequences = []
    for sequence in sequences:
        new_sequence = []
        first_items.append(sequence[0])
        for i in range(len(sequence) - 1):
            cur_item = sequence[i]
            next_item = sequence[i + 1]
            new_item = [0, 0]
            new_item[0] = (next_item[0] - cur_item[0]) / (next_item[0] + eps)
            new_item[1] = (next_item[1] - cur_item[1]) / (next_item[1] + eps)
            new_sequence.append(new_item)
        new_sequences.append(new_sequence)
    return first_items, new_sequences


def get_diff(sequences):
    """
        把 generate_data_pairs 生成的 all_inputs 或者 all_targets 转化为 差值
        :return:
            first_items : list, sequences 里 每个 sequence 的第一个数据 的列表（用于数据还原）
            new_sequences : list, 转化为 差值 之后的 sequences
        """
    first_items = []
    new_sequences = []
    for sequence in sequences:
        new_sequence = []
        first_items.append(sequence[0])
        for i in range(len(sequence) - 1):
            cur_item = sequence[i]
            next_item = sequence[i + 1]
            new_item = [0, 0]
            new_item[0] = next_item[0] - cur_item[0]
            new_item[1] = next_item[1] - cur_item[1]
            new_sequence.append(new_item)
        new_sequences.append(new_sequence)
    return first_items, new_sequences


def unget_rise(first_items, sequences):
    """
    与 get_rise 做相反的操作
    :param first_items: list, sequences 里 每个 sequence 的第一个数据 的列表
    :param sequences: list, 转化为波动率之后的 sequences
    :return:
    """
    eps = 1e-11
    new_sequences = []
    for i, sequence in enumerate(sequences):
        new_sequence = []
        new_sequence.append(first_items[i])
        last_item = first_items[i]
        for j in range(len(sequence)):
            cur_rise = sequence[j]
            cur_item = [0, 0]
            cur_item[0] = (eps * cur_rise[0] + last_item[0]) / (1 - cur_rise[0])
            cur_item[1] = (eps * cur_rise[1] + last_item[1]) / (1 - cur_rise[1])
            new_sequence.append(cur_item)
            last_item = cur_item
        new_sequences.append(new_sequence)
    return new_sequences


def unget_diff(first_items, sequences):
    """
    与 get_rise 做相反的操作
    :param first_items: list, sequences 里 每个 sequence 的第一个数据 的列表
    :param sequences: list, 转化为波动率之后的 sequences
    :return:
    """
    new_sequences = []
    for i, sequence in enumerate(sequences):
        new_sequence = []
        new_sequence.append(first_items[i])
        last_item = first_items[i]
        for j in range(len(sequence)):
            cur_rise = sequence[j]
            cur_item = [0, 0]
            cur_item[0] = cur_rise[0] + last_item[0]
            cur_item[1] = cur_rise[1] + last_item[1]
            new_sequence.append(cur_item)
            last_item = cur_item
        new_sequences.append(new_sequence)
    return new_sequences


def test_get_rise():
    sequences = [
        [[0, 0], [1, 1], [2, 2]],
        [[1, 2], [4, 5], [6, 7]],
        [[3, 4], [7, 8]]
    ]
    first_items, new_sequences = get_rise(sequences)
    print(sequences)
    print(first_items)
    print(new_sequences)
    print("---")
    print(unget_rise(first_items, new_sequences))


def test_get_diff():
    sequences = [
        [[0, 0], [1, 1], [2, 2]],
        [[1, 2], [4, 5], [6, 7]],
        [[3, 4], [7, 8]]
    ]
    first_items, new_sequences = get_diff(sequences)
    print(sequences)
    print(first_items)
    print(new_sequences)
    print("---")
    print(unget_diff(first_items, new_sequences))


def padding_variable_length_sequence(all_sequences):
    """
    generate_data_pairs 函数 生成的 all_inputs 和 all_target 是变长的序列
    使用这个方法，可以把他们打包为一个 用0填充的 Tensor  ( 方便神经网络并行计算 )
    example：
        打包前：
            # all_inputs
            [[0, 1, 2, 3, 4, 5, 6],
            [7, 7],
            [6, 8]]
            # all_target
            [[1, 2, 3, 3, 3, 1, 4],
            [5, 5],
            [4, 5]]
        打包后：
            # all_inputs
            array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.],
                   [ 8.,  8.,  0.,  0.,  0.,  0.,  0.],
                   [ 7.,  9.,  0.,  0.,  0.,  0.,  0.]])
            # all_target
            array([[ 1.,  2.,  3.,  3.,  3.,  1.,  4.],
                   [ 5.,  5.,  0.,  0.,  0.,  0.,  0.],
                   [ 4.,  5.,  0.,  0.,  0.,  0.,  0.]])
    :param all_sequences: 可以是 all_inputs 或者 all_target
    :return:
        array : torch.Tensor, 打包后的数组
        lengths : torch.Tensor, 每个序列的长度
    """
    # 计算每个序列的长度
    lengths = [len(sequence) for sequence in all_sequences]
    lengths = torch.Tensor(lengths).long()
    sequences = [torch.Tensor(sequence) for sequence in all_sequences]
    array = pad_sequence(sequences, batch_first=True, padding_value=-100)
    return array, lengths


def unpadding_variable_length_sequence(array: torch.Tensor, lengths):
    """
    与 padding_variable_length_sequence 做相反的过程
    :param array: torch.Tensor, 打包后的数组
    :param lengths: torch.Tensor, 每个序列的长度
    :return: list, 还原后的序列
    """
    lengths = lengths.detach().cpu().numpy().tolist()
    array = array.detach().cpu().numpy()
    all_sequences = []
    batch_size, max_len, input_size = array.shape
    for i in range(batch_size):
        all_sequences.append(array[i, 0:int(lengths[i])].tolist())
    return all_sequences


def get_acc(random_circles, predict_targets, targets):
    """
    计算准确度
    准确度判断方法：
            以目标点为圆心，间隔1度的点的直线为直径的圆内
    :param random_circles: list,
    :param predict_targets: list, 每一项是一个预测出的点
    :param targets: list, 每一项是标签点
    :return:
    """
    nb_correct = 0
    for i in range(len(random_circles)):
        # 计算准确度允许的误差距离 （圆上 转过1度 的直线距离 的 二分之一）
        target_distance = random_circles[i].radius * math.sin(angle_to_radian(1 / 2))
        distance = (predict_targets[i][0] - targets[i][0]) ** 2  # (x1-x2)^2
        distance += (predict_targets[i][1] - targets[i][1]) ** 2  # (y1-y2)^2
        # 预测出的点 与目标点的距离
        distance = math.sqrt(distance)
        if distance <= target_distance:
            nb_correct += 1
    return nb_correct / len(random_circles)


if __name__ == "__main__":
    test_plot_data_pairs()
