"""可选对照方法；不参与主程序的 MEA 定级。"""
import numpy as np


def membership(value, intervals):
    """在等级区间中心之间线性插值；返回与输入等级顺序一致的隶属度。"""
    centers = np.mean(intervals, axis=1)
    order = np.argsort(centers)
    # 端部取肩形隶属，中间仅相邻两级分配权重。
    return np.array([np.interp(value, centers[order], basis[order]) for basis in np.eye(len(order))])
