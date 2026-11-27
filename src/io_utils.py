"""读入三张独立数据表，按编号对齐并验证项目约束。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRADES = ['病态', '一般', '中等', '良好', '优良']
YEARS = [str(year) for year in range(2020, 2025)]


def require(condition, message):
    """校验失败时给出可操作的异常，不依赖 assert 开关。"""
    if not condition:
        raise ValueError(message)


def load_data(folder=ROOT / 'data'):
    """返回对齐且通过校验的观测、权重及经典域表。"""
    tables = []
    expected = [f'X{num}' for num in range(1, 28)]
    for name in ['indicators_2020_2024', 'weights', 'thresholds']:
        table = pd.read_csv(Path(folder) / f'{name}.csv', index_col='编号')
        require(table.index.is_unique and set(table.index) == set(expected), f'{name}: 编号不完整或重复')
        tables.append(table.loc[expected].copy())
    data, weights, thresholds = tables
    for field in ['名称', '方向', '原权重', 'exclude']:
        require(data[field].equals(weights[field]), f'指标表和权重表的{field}不一致')
    require(weights.exclude.dtype == bool and set(weights.index[weights.exclude]) == {'X16', 'X23'}, '剔除项错误')
    require(weights['方向'].isin(['正', '负']).all(), '未知方向')
    require(set(weights['分组']) == {'活力', '组织结构', '恢复力', '服务功能', '人群健康'}, '分组不完整')
    raw = weights['原权重']
    require(np.isfinite(raw).all() and (raw > 0).all(), '原权重须为正有限数')
    normalized = raw.where(~weights.exclude, 0) / raw[~weights.exclude].sum()
    require(np.allclose(weights['归一化权重'], normalized, atol=1e-12, rtol=0), '归一化权重不符')
    require(thresholds['阈值策略'].isin(['Table3', '样本等宽']).all(), '未知阈值策略')
    for key, row in thresholds.iterrows():
        intervals = parse_intervals(row)
        ordered = intervals[np.argsort(intervals[:, 0])]
        require(np.isfinite(ordered).all() and (ordered[:, 1] > ordered[:, 0]).all()
                and np.array_equal(ordered[:-1, 1], ordered[1:, 0]), f'{key}: 经典域非连续')
        if row['阈值策略'] == 'Table3':
            sign = 1 if weights.at[key, '方向'] == '正' else -1
            require((sign * np.diff(intervals[:, 0]) > 0).all(), f'{key}: 方向与原阈值不一致')
    active = data.loc[~weights.exclude, YEARS].apply(pd.to_numeric, errors='raise')
    require(np.isfinite(active).all().all(), '保留项有缺测或无穷值')
    for flags in data['假设年份'].dropna():
        require(set(str(flags).split(';')) <= set(YEARS), '假设年份无效')
    return data, weights, thresholds


def parse_intervals(row):
    """按病态至优良顺序解析原表区间，保留原始方向。"""
    return np.array([[float(end) for end in row[grade].split('–')] for grade in GRADES])


def write_csv(frame, path):
    """输出 UTF-8 BOM，方便中文 Excel 直接打开。"""
    frame.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n')
