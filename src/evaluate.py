"""南京城市生态系统健康物元评价，可从任意目录直接运行。"""
import numpy as np
import pandas as pd

if __package__:
    from .io_utils import GRADES, ROOT, YEARS, load_data, parse_intervals, require, write_csv
else:
    from io_utils import GRADES, ROOT, YEARS, load_data, parse_intervals, require, write_csv


def correlations(value, intervals, direction):
    """输入无序原量纲区间；按方向排列等级，返回五级关联度及是否截界。"""
    require(direction in ('正', '负'), '未知方向')
    bounds = intervals[np.argsort(intervals[:, 0])]
    if direction == '负':
        bounds = bounds[::-1]
    lower, upper = bounds.T
    used = np.clip(value, lower.min(), upper.max())
    # 到闭区间的有符号距离；内部为负，边界为零，外部为正。
    distance = np.maximum(lower - used, used - upper)
    joint = max(lower.min() - used, used - upper.max())
    scores = -distance / (upper - lower)
    outside = distance > 0
    scores[outside] = distance[outside] / (joint - distance[outside])
    return scores, bool(used != value)


def indicator_scores(data, weights, thresholds):
    """返回每个保留项每年的关联度，标明域来源、越界与假设。"""
    records = []
    for key, weight in weights.loc[~weights.exclude].iterrows():
        series = data.loc[key, YEARS].astype(float)
        policy = thresholds.at[key, '阈值策略']
        if policy == '样本等宽':
            require(series.min() < series.max(), f'{key}: 常数样本不能等宽分级')
            edges = np.linspace(series.min(), series.max(), len(GRADES) + 1)
            intervals = np.column_stack((edges[:-1], edges[1:]))
        else:
            intervals = parse_intervals(thresholds.loc[key])
        assumed = str(data.at[key, '假设年份']).split(';')
        for year, value in series.items():
            scores, clipped = correlations(value, intervals, weight['方向'])
            records.append({'编号': key, '名称': weight['名称'], '分组': weight['分组'], '年份': year,
                            '值': value, '方向': weight['方向'], '阈值策略': policy, '越界': clipped,
                            'is_assumed': year in assumed, '原权重': weight['原权重'],
                            **dict(zip(GRADES, scores)), '判定等级': GRADES[scores.argmax()]})
    return pd.DataFrame(records)


def summarize(scores):
    """分别按阈值口径和假设值情景汇总；组内和年内分别归一化。"""
    records = []
    for year, annual in scores.groupby('年份'):
        for policy, eligible in [('25项探索', annual), ('仅原经典域', annual[annual['阈值策略'] == 'Table3'])]:
            for scenario, rows in [('含假设值', eligible), ('剔除假设值', eligible[~eligible.is_assumed])]:
                values = np.average(rows[GRADES], axis=0, weights=rows['原权重'])
                record = {'年份': year, '阈值口径': policy, '情景': scenario, '参与项数': len(rows),
                          '假设值项数': rows.is_assumed.sum(), '原权重合计': rows['原权重'].sum(),
                          **dict(zip(GRADES, values)), '判定等级': GRADES[values.argmax()]}
                for group, subset in rows.groupby('分组'):
                    group_scores = np.average(subset[GRADES], axis=0, weights=subset['原权重'])
                    record[f'{group}_得分'] = group_scores.max()
                    record[f'{group}_等级'] = GRADES[group_scores.argmax()]
                records.append(record)
    return pd.DataFrame(records)


def run(data_dir=ROOT / 'data', output_dir=ROOT / 'outputs'):
    """读取输入、写出三张结果表，返回年度表。"""
    data, weights, thresholds = load_data(data_dir)
    scores = indicator_scores(data, weights, thresholds)
    annual = summarize(scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scores, output_dir / 'scores_by_indicator.csv')
    write_csv(annual, output_dir / 'result_annual.csv')
    write_csv(weights.reset_index(), output_dir / 'weight_check.csv')
    return annual


if __name__ == '__main__':
    print(run().pivot(index='年份', columns=['阈值口径', '情景'], values='判定等级').to_string())
