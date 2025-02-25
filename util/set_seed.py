import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    设置全局随机种子以确保结果可复现。
    
    参数:
    - seed (int): 要设置的随机种子，默认值为42。
    """
    # Python内置随机库
    random.seed(seed)
    
    # NumPy库
    np.random.seed(seed)
    
    # PyTorch（CPU和GPU）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时使用
    
    # PyTorch的确定性和性能设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")