import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

__all__ = [
    'data_pre_processing',
    'prob_prior',
    'kernel_func',
    'p_density_train',
    'p_density_test',
    'prob_density_train',
    'prob_density_test',
    'prob_posterior',
    'model',
    'bayesian_classifier',
]


def data_pre_processing(df: pd.DataFrame):
    """数据预处理

    Args:
        df (pd.DataFrame): 原生数据集

    Returns:
        tuple: patterns_train, patterns_valid, classes_train, classes_valid
    """
    # 类别编码
    series = df[df.columns[-1]].value_counts()
    dict = {series.index[i]: i for i in range(series.size)}
    df[df.columns[-1]] = df[df.columns[-1]].map(dict)
    # 划分数据集
    patterns_train, patterns_valid, classes_train, classes_valid = train_test_split(
        df.iloc[:, :-1].values, df.iloc[:, -1].values, test_size=0.2)
    # 特征缩放
    st_patterns = StandardScaler()
    patterns_train = st_patterns.fit_transform(patterns_train)
    patterns_valid = st_patterns.transform(patterns_valid)
    # 数据类型转换
    patterns_train = torch.Tensor(patterns_train)
    patterns_valid = torch.Tensor(patterns_valid)
    classes_train = torch.Tensor(classes_train)
    classes_valid = torch.Tensor(classes_valid)

    return patterns_train, patterns_valid, classes_train, classes_valid


def prob_prior(df_train: pd.DataFrame) -> torch.Tensor:
    """先验概率

    Args:
        df_train (pd.DataFrame)

    Returns:
        torch.Tensor: P_prior
    """

    series = df_train[df_train.columns[-1]].value_counts(normalize=True)
    P_prior = torch.Tensor([series.values[i] for i in range(series.size)])

    return P_prior


def kernel_func(X: torch.Tensor, Y: torch.Tensor, h: torch.Tensor,
                kernel: str) -> torch.Tensor:
    """核函数       

    Args:
        X (torch.Tensor)
        Y (torch.Tensor)
        h (torch.Tensor): 核函数带宽
        kernel (str): 核函数类型, 包括 'gaussian', 'epanechnikov', 'uniform' 三种

    Raises:
        ValueError: Invalid kernel type

    Returns:
        torch.Tensor
    """

    distance = torch.abs(X - Y)
    if kernel == 'gaussian':
        return torch.exp(-distance**2 / (2 * h**2))
    elif kernel == 'epanechnikov':
        return 3 / 4 * (1 - ((distance <= h) * distance)**2 / h**2)
    elif kernel == 'uniform':
        return (distance <= h) * distance / (2 * h)
    else:
        raise ValueError('Invalid kernel type')


def p_density_train(patterns: torch.Tensor, h: torch.Tensor,
                    kernel: str) -> torch.Tensor:
    """训练集概率密度估计函数

    Args:
        patterns (torch.Tensor)
        h (torch.Tensor): 核函数带宽
        kernel (str): 核函数类型, 包括 'gaussian', 'epanechnikov', 'uniform' 三种

    Returns:
        torch.Tensor: p_train
    """
    # 核密度估计
    p_train = kernel_func(patterns[:, None, :].permute(2, 1, 0),
                          patterns.permute(1, 0)[..., None], h,
                          kernel).sum(dim=1) / patterns.shape[0]
    # 朴素贝叶斯方法
    p_train = p_train.prod(dim=0)

    return p_train


def p_density_test(patterns_test: torch.Tensor, patterns_train: torch.Tensor,
                   p_density_train: torch.Tensor) -> torch.Tensor:
    """测试集概率密度估计函数

    Args:
        patterns_test (torch.Tensor)
        patterns_train (torch.Tensor)
        p_density_train (torch.Tensor)

    Returns:
        torch.Tensor: p_test
    """

    # 点积相似性
    comparability = torch.mm(patterns_test, patterns_train.T) / torch.pow(
        patterns_train.T, 2).sum(dim=0)[None, :]
    # 最大相似性概率
    max_ratios, max_indices = torch.max(comparability, dim=1)
    p_test = p_density_train[max_indices] * max_ratios

    return p_test


def prob_density_train(patterns_train: torch.Tensor, list_patterns_class: list,
                       h: torch.Tensor, kernel: str) -> tuple:
    """训练集概率密度

    Args:
        patterns_train (torch.Tensor)
        list_patterns_class (list)
        h (torch.Tensor): 核函数带宽
        kernel (str): 核函数类型, 包括 'gaussian', 'epanechnikov', 'uniform' 三种

    Returns:
        tuple: p_sample_train, list_p_class_train
    """

    p_sample_train = p_density_train(patterns_train, h, kernel)
    list_p_class_train = [
        p_density_train(list_patterns_class[i], h, kernel)
        for i in range(len(list_patterns_class))
    ]

    return p_sample_train, list_p_class_train


def prob_density_test(patterns_test: torch.Tensor,
                      patterns_train: torch.Tensor, list_patterns_class: list,
                      p_sample_train: torch.Tensor,
                      list_p_class_train: list) -> tuple:
    """测试集概率密度

    Args:
        patterns_test (torch.Tensor)
        patterns_train (torch.Tensor)
        list_patterns_class (list)
        p_sample_train (torch.Tensor)
        list_p_class_train (list)

    Returns:
        tuple: p_sample_test, p_class_test
    """

    p_sample_test = p_density_test(patterns_test, patterns_train,
                                   p_sample_train)
    p_class_test = p_density_test(patterns_test, list_patterns_class[0],
                                  list_p_class_train[0])
    for i in range(len(list_patterns_class) - 1):
        p_class_test = torch.cat([
            p_class_test[:, None],
            p_density_test(patterns_test, list_patterns_class[i + 1],
                           list_p_class_train[i + 1])[:, None]
        ],
                                 dim=1)

    return p_sample_test, p_class_test


def prob_posterior(patterns_train: torch.Tensor, patterns_test: torch.Tensor,
                   classes_train: torch.Tensor, h: torch.Tensor,
                   kernel: str) -> torch.Tensor:
    """后验概率

    Args:
        patterns_train (torch.Tensor)
        patterns_test (torch.Tensor)
        classes_train (torch.Tensor)
        h (torch.Tensor): 核函数带宽
        kernel (str): 核函数类型, 包括 'gaussian', 'epanechnikov', 'uniform' 三种

    Returns:
        torch.Tensor: P_posterior
    """

    df_train = pd.DataFrame(
        torch.cat([patterns_train, classes_train[:, None]],
                  dim=1).detach().numpy())
    series = df_train[df_train.columns[-1]].value_counts()
    list_patterns_class = [
        torch.Tensor(df_train.loc[df_train[df_train.columns[-1]] ==
                                  series.index[i]].iloc[:, :-1].values)
        for i in range(series.size)
    ]

    P_prior = prob_prior(df_train)
    p_sample_train, list_p_class_train = prob_density_train(
        patterns_train, list_patterns_class, h, kernel)
    p_sample_test, p_class_test = prob_density_test(patterns_test,
                                                    patterns_train,
                                                    list_patterns_class,
                                                    p_sample_train,
                                                    list_p_class_train)

    P_prior = P_prior[None, :]
    p_sample_test = p_sample_test[:, None]
    P_posterior = torch.Tensor(P_prior * p_class_test / p_sample_test)
    P_posterior = torch.divide(P_posterior, P_posterior.sum(dim=1)[:, None])

    return P_posterior


def model(patterns_train: torch.Tensor, patterns_test: torch.Tensor,
          classes_train: torch.Tensor, h: torch.Tensor,
          kernel: str) -> torch.Tensor:
    """贝叶斯模型

    Args:
        patterns_train (torch.Tensor)
        patterns_test (torch.Tensor)
        classes_train (torch.Tensor)
        h (torch.Tensor): 核函数带宽
        kernel (str): 核函数类型, 包括 'gaussian', 'epanechnikov', 'uniform' 三种

    Returns:
        torch.Tensor: prob_results_classes, results_classes
    """

    P_posterior = prob_posterior(patterns_train, patterns_test, classes_train,
                                 h, kernel)

    prob_results_classes, results_classes = torch.max(P_posterior, dim=1)

    return prob_results_classes, results_classes


def bayesian_classifier(df: pd.DataFrame,
                        h: torch.Tensor,
                        kernel: str = 'gaussian',
                        mode: str = 'train',
                        num_epochs: int = 300,
                        num_search: int = 20,
                        step_search: int = 0.025,
                        patterns_test: torch.Tensor = None) -> tuple:
    """贝叶斯分类器

    Args:
        df (pd.DataFrame): 原生数据集
        h (torch.Tensor): 核函数带宽
        kernel (str, optional): 核函数类型, 包括 'gaussian', 'epanechnikov', 'uniform' 三种. Defaults to 'gaussian'.
        mode (str, optional): 分类器模式, 包括 'train', 'eval' 两种. Defaults to 'train'.
        num_epochs (int, optional): 训练轮数. Defaults to 300.
        num_search (int, optional): 参数搜索数. Defaults to 20.
        step_search (int, optional): 参数搜索步长. Defaults to 0.025.
        patterns_test (torch.Tensor, optional): 需要进行的预测的测试集. Defaults to None.

    Raises:
        TypeError: patterns_test 类型要求为 torch.Tensor
        ValueError: patterns_test 的 pattern 数量要求与 df 的 pattern 数量匹配
        ValueError: patterns_test 的 pattern 数量要求与 df 的 pattern 数量匹配
        ValueError: mode 值非法

    Returns:
        tuple: 当 mode 为 'train' 时, 返回 accuracy, h; 当 mode 为 'eval' 时, 返回 prob_results_classes, results_classes
    """
    # 训练
    if mode == 'train':
        # 暴力搜索优化参数
        accr = []
        for i in range(num_search):
            for _ in range(num_epochs):
                patterns_train, patterns_valid, classes_train, classes_valid = data_pre_processing(
                    df)
                accr.append(
                    accuracy_score(
                        classes_valid,
                        model(patterns_train, patterns_valid, classes_train,
                              h + step_search * i, kernel)[1]))
        accr = torch.tensor(accr, dtype=torch.float32).reshape(num_search, -1)
        accuracy, index = accr.sum(dim=1).divide(num_epochs).max(dim=0)
        h += index * step_search
        return accuracy, h
    # 估计
    elif mode == 'eval':
        if type(patterns_test) != torch.Tensor:
            raise TypeError(
                f'\'patterns_test\' must be a torch.Tensor but not be {type(patterns_test)}, when \'mode\' is \'eval\''
            )
        else:
            if (len(patterns_test.shape)
                    == 1) & (patterns_test.shape[0] != df.columns.size - 1):
                raise ValueError(
                    f'\'patterns_test.shape[0]\' must be equivalent to number of patterns in \'df\''
                )
            elif (len(patterns_test.shape)
                  == 2) & (patterns_test.shape[1] != df.columns.size - 1):
                raise ValueError(
                    f'\'patterns_test.shape[1]\' must be equivalent to number of patterns in \'df\''
                )
            else:
                patterns_train, patterns_valid, classes_train, classes_valid = data_pre_processing(
                    df)
                prob_results_classes, results_classes = model(
                    patterns_train, patterns_test, classes_train, h, kernel)
                return prob_results_classes, results_classes
    else:
        raise ValueError('Invalid mode type')


# 测试程序
def main():
    # 数据加载
    data_file = os.path.join('..', 'data', 'data_salmonbass.xlsx')
    data_raw = pd.read_excel(data_file)
    # 初始化模型参数
    h = torch.tensor(0.1, dtype=torch.float32)
    # 训练
    accuracy, h = bayesian_classifier(data_raw, h)
    print(accuracy, '\n', h)
    # 估计
    probabilities, results = bayesian_classifier(
        data_raw,
        h,
        mode='eval',
        patterns_test=torch.Tensor(data_raw.head(10).iloc[:, :-1].values))
    print(probabilities, '\n', results)

    return 0


if __name__ == "__main__":
    main()