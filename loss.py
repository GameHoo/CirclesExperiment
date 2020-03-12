import torch
from circles_experiment.data_process import padding_variable_length_sequence, unpadding_variable_length_sequence, \
    get_rise, unget_rise, get_diff, unget_diff
import torch.nn as nn

mseloss = nn.MSELoss()


def loss1(model, all_inputs, all_targets, device='cpu', get_predict_target=True):
    """
    每个时间步计算loss
    :param model: 神经网络模型
    :param all_inputs: list, 每一项是一个输入，每个输入是一个坐标序列。
    例如：
        [
            [[1,1],[2,2]]
            [1,2],[3,4],[5,6]]
        ]
        ）
    :param all_targets: list, 每一项是一个目标输出，每个输出是一个坐标序列
    :param device: 计算用的设备
    :param get_predict_target: 是否输出预测的值 （用来计算准确率）
    :return:
        loss: torch.Tensor, loss 值
        predict_targets: list, 神经网络预测出的坐标值
    """
    # 转化为 tensor
    input_array, lengths = padding_variable_length_sequence(all_inputs)
    output_array, _ = padding_variable_length_sequence(all_targets)
    input_array, output_array = input_array.to(device), output_array.to(device)
    # 神经网络预测
    mask = (input_array != -100).float()
    predict_output_array = model(input_array, lengths)
    # 计算 loss (用 mask 屏蔽掉 无效数据的 loss)
    output_array *= mask
    predict_output_array *= mask
    loss = mseloss(output_array, predict_output_array)
    predict_targets = None
    if get_predict_target:
        # 计算实际预测值
        predict_targets = unpadding_variable_length_sequence(predict_output_array, lengths)
        predict_targets = [target[-1] for target in predict_targets]
    return loss, predict_targets


def loss2(model, all_inputs, all_targets, device='cpu', get_predict_target=True):
    """
    只在最后一个时间步计算loss
    :param model: 神经网络模型
    :param all_inputs: list, 每一项是一个输入，每个输入是一个坐标序列。
    例如：
        [
            [[1,1],[2,2]]
            [1,2],[3,4],[5,6]]
        ]
        ）
    :param all_targets: list, 每一项是一个目标输出，每个输出是一个坐标序列
    :param device: 计算用的设备
    :param get_predict_target: 是否输出预测的值 （用来计算准确率）
    :return:
        loss: torch.Tensor, loss 值
        predict_targets: list, 神经网络预测出的坐标值
    """
    # 转化为 tensor
    input_array, lengths = padding_variable_length_sequence(all_inputs)
    output_array, _ = padding_variable_length_sequence(all_targets)
    input_array, output_array = input_array.to(device), output_array.to(device)
    lengths = lengths.to(device)
    # 神经网络预测
    predict_output_array = model(input_array, lengths)
    # 计算 loss
    # 只取最后一个
    _lengths = lengths - 1  # 最后一个数据的 索引
    _lengths = _lengths.view(-1, 1, 1)
    _lengths = _lengths.expand(-1, 1, 2)
    output_array_last = output_array.gather(1, _lengths)
    predict_output_array_last = predict_output_array.gather(1, _lengths)
    loss = mseloss(output_array_last, predict_output_array_last)
    predict_targets = None
    if get_predict_target:
        # 计算实际预测值
        predict_targets = unpadding_variable_length_sequence(predict_output_array, lengths)
        predict_targets = [target[-1] for target in predict_targets]
    return loss, predict_targets


def loss3(model, all_inputs, all_targets, device='cpu', get_predict_target=True):
    """
    计算波动率之后，每个时间步计算loss
    :param model: 神经网络模型
    :param all_inputs: list, 每一项是一个输入，每个输入是一个坐标序列。
    例如：
        [
            [[1,1],[2,2]]
            [1,2],[3,4],[5,6]]
        ]
        ）
    :param all_targets: list, 每一项是一个目标输出，每个输出是一个坐标序列
    :param device: 计算用的设备
    :param get_predict_target: 是否输出预测的值 （用来计算准确率）
    :return:
        loss: torch.Tensor, loss 值
        predict_targets: list, 神经网络预测出的坐标值
    """
    # 计算波动率
    inputs_first_items, all_inputs_rise = get_rise(all_inputs)
    targets_first_items, all_targets_rise = get_rise(all_targets)
    # 转化为 tensor
    input_rise_array, rise_lengths = padding_variable_length_sequence(all_inputs_rise)
    output_rise_array, _ = padding_variable_length_sequence(all_targets_rise)
    input_rist_array, output_rise_array = input_rise_array.to(device), output_rise_array.to(device)
    rise_lengths = rise_lengths.to(device)
    # 神经网络预测
    mask = (input_rist_array != -100).float()
    predict_output_rise_array = model(input_rist_array, rise_lengths)
    # 计算 波动率的 loss (用 mask 屏蔽掉 无效数据的 loss)
    output_rise_array *= mask
    predict_output_rise_array *= mask
    loss = mseloss(output_rise_array, predict_output_rise_array)
    predict_targets = None
    if get_predict_target:
        # 计算实际预测值
        predict_rise_targets = unpadding_variable_length_sequence(predict_output_rise_array, rise_lengths)
        predict_targets = unget_rise(targets_first_items, predict_rise_targets)
        predict_targets = [target[-1] for target in predict_targets]
    return loss, predict_targets


def loss4(model, all_inputs, all_targets, device='cpu', get_predict_target=True):
    """
    计算波动率之后，最后一个时间步计算loss
    :param model: 神经网络模型
    :param all_inputs: list, 每一项是一个输入，每个输入是一个坐标序列。
    例如：
        [
            [[1,1],[2,2]]
            [1,2],[3,4],[5,6]]
        ]
        ）
    :param all_targets: list, 每一项是一个目标输出，每个输出是一个坐标序列
    :param device: 计算用的设备
    :param get_predict_target: 是否输出预测的值 （用来计算准确率）
    :return:
        loss: torch.Tensor, loss 值
        predict_targets: list, 神经网络预测出的坐标值
    """
    # 计算波动率
    inputs_first_items, all_inputs_rise = get_rise(all_inputs)
    targets_first_items, all_targets_rise = get_rise(all_targets)
    # 转化为 tensor
    input_rise_array, rise_lengths = padding_variable_length_sequence(all_inputs_rise)
    output_rise_array, _ = padding_variable_length_sequence(all_targets_rise)
    input_rist_array, output_rise_array = input_rise_array.to(device), output_rise_array.to(device)
    rise_lengths = rise_lengths.to(device)
    # 神经网络预测
    predict_output_rise_array = model(input_rist_array, rise_lengths)
    # 计算 波动率的 loss
    # 只取最后一个
    _lengths = rise_lengths - 1  # 最后一个数据的索引
    _lengths = _lengths.view(-1, 1, 1)
    _lengths = _lengths.expand(-1, 1, 2)
    output_rise_array_last = output_rise_array.gather(1, _lengths)
    predict_output_rise_array_last = predict_output_rise_array.gather(1, _lengths)
    loss = mseloss(output_rise_array_last, predict_output_rise_array_last)
    predict_targets = None
    if get_predict_target:
        # 计算实际预测值
        predict_rise_targets = unpadding_variable_length_sequence(predict_output_rise_array, rise_lengths)
        predict_targets = unget_rise(targets_first_items, predict_rise_targets)
        predict_targets = [target[-1] for target in predict_targets]
    return loss, predict_targets


def loss5(model, all_inputs, all_targets, device='cpu', get_predict_target=True):
    """
    计算 差值 之后，每个时间步计算loss
    :param model: 神经网络模型
    :param all_inputs: list, 每一项是一个输入，每个输入是一个坐标序列。
    例如：
        [
            [[1,1],[2,2]]
            [1,2],[3,4],[5,6]]
        ]
        ）
    :param all_targets: list, 每一项是一个目标输出，每个输出是一个坐标序列
    :param device: 计算用的设备
    :param get_predict_target: 是否输出预测的值 （用来计算准确率）
    :return:
        loss: torch.Tensor, loss 值
        predict_targets: list, 神经网络预测出的坐标值
    """
    # 计算差值
    inputs_first_items, all_inputs_diff = get_diff(all_inputs)
    targets_first_items, all_targets_diff = get_diff(all_targets)
    # 转化为 tensor
    input_diff_array, diff_lengths = padding_variable_length_sequence(all_inputs_diff)
    output_diff_array, _ = padding_variable_length_sequence(all_targets_diff)
    input_diff_array, output_diff_array = input_diff_array.to(device), output_diff_array.to(device)
    diff_lengths = diff_lengths.to(device)
    # 神经网络预测
    mask = (input_diff_array != -100).float()
    predict_output_diff_array = model(input_diff_array, diff_lengths)
    # 计算 差值的 loss (用 mask 屏蔽掉 无效数据的 loss)
    output_diff_array *= mask
    predict_output_diff_array *= mask
    loss = mseloss(output_diff_array, predict_output_diff_array)
    predict_targets = None
    if get_predict_target:
        # 计算实际预测值
        predict_diff_targets = unpadding_variable_length_sequence(predict_output_diff_array, diff_lengths)
        predict_targets = unget_diff(targets_first_items, predict_diff_targets)
        predict_targets = [target[-1] for target in predict_targets]
    return loss, predict_targets


def loss6(model, all_inputs, all_targets, device='cpu', get_predict_target=True):
    """
    计算 差值 之后，最后一个时间步计算loss
    :param model: 神经网络模型
    :param all_inputs: list, 每一项是一个输入，每个输入是一个坐标序列。
    例如：
        [
            [[1,1],[2,2]]
            [1,2],[3,4],[5,6]]
        ]
        ）
    :param all_targets: list, 每一项是一个目标输出，每个输出是一个坐标序列
    :param device: 计算用的设备
    :param get_predict_target: 是否输出预测的值 （用来计算准确率）
    :return:
        loss: torch.Tensor, loss 值
        predict_targets: list, 神经网络预测出的坐标值
    """
    # 计算差值
    inputs_first_items, all_inputs_diff = get_diff(all_inputs)
    targets_first_items, all_targets_diff = get_diff(all_targets)
    # 转化为 tensor
    input_diff_array, diff_lengths = padding_variable_length_sequence(all_inputs_diff)
    output_diff_array, _ = padding_variable_length_sequence(all_targets_diff)
    input_diff_array, output_diff_array = input_diff_array.to(device), output_diff_array.to(device)
    diff_lengths = diff_lengths.to(device)
    # 神经网络预测
    predict_output_diff_array = model(input_diff_array, diff_lengths)
    # 计算 差值的 loss
    # 只取最后一个
    _lengths = diff_lengths - 1  # 最后一个数据的索引
    _lengths = _lengths.view(-1, 1, 1)
    _lengths = _lengths.expand(-1, 1, 2)
    output_diff_array_last = output_diff_array.gather(1, _lengths)
    predict_output_diff_array_last = predict_output_diff_array.gather(1, _lengths)
    loss = mseloss(output_diff_array_last, predict_output_diff_array_last)
    predict_targets = None
    if get_predict_target:
        # 计算实际预测值
        predict_diff_targets = unpadding_variable_length_sequence(predict_output_diff_array, diff_lengths)
        predict_targets = unget_diff(targets_first_items, predict_diff_targets)
        predict_targets = [target[-1] for target in predict_targets]
    return loss, predict_targets
