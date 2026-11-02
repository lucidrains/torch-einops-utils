import torch

from torch_einops_utils import format_table, log_table


def test_format_table_single_row():
    line = format_table(
        ep = 200,
        reward = 187.04,
        steps = 1000,
        avg100 = 66.03,
        formatters = dict(reward = '.2f', avg100 = '.2f')
    )

    assert line == 'ep 200  reward 187.04  steps 1000  avg100 66.03'


def test_format_table_multiple_rows():
    rows = [
        dict(ep = 200, reward = 187.04, steps = 1000, avg100 = 66.03),
        dict(ep = 225, reward = 76.72, steps = 1000, avg100 = 89.98)
    ]

    text = format_table(rows, formatters = dict(reward = '.2f', avg100 = '.2f'))

    assert text == '\n'.join([
        'ep   reward  steps  avg100',
        '200  187.04   1000   66.03',
        '225   76.72   1000   89.98'
    ])


def test_format_table_columns_and_header():
    rows = [
        dict(a = 1, b = 2, c = 3),
        dict(a = 4, b = 5, c = 6)
    ]

    text = format_table(rows, columns = ['c', 'a'], header = False)

    assert text == '3  1\n6  4'


def test_format_table_tensor_and_formatters():
    line = format_table(
        dict(loss = torch.tensor(1.23456), acc = 0.98765, step = 1),
        formatters = dict(loss = '.2f', acc = '.1%')
    )

    assert line == 'loss 1.23  acc 98.8%  step 1'


def test_format_table_title():
    text = format_table(dict(a = 1), title = 'config')

    assert text == 'config\na 1'


def test_log_table(capsys):
    returned = log_table(ep = 200, reward = 187.04, formatters = dict(reward = '.2f'))

    captured = capsys.readouterr()
    assert captured.out == 'ep 200  reward 187.04\n'
    assert returned == 'ep 200  reward 187.04'
