"""以实际输入和可手算的小例验证项目口径与关联函数。"""
import unittest

import numpy as np

from src.evaluate import correlations, indicator_scores, summarize
from src.fuzzy import membership
from src.io_utils import GRADES, load_data


class SmokeTest(unittest.TestCase):
    def test_weights(self):
        """确认 27 项、剔除规则及有效权重。"""
        data, weights, _ = load_data()
        self.assertEqual(len(data), 27)
        self.assertEqual(set(weights.index[weights.exclude]), {'X16', 'X23'})
        self.assertEqual(weights.loc[weights.exclude, '归一化权重'].sum(), 0)
        self.assertAlmostEqual(weights['归一化权重'].sum(), 1)
        self.assertTrue((data.loc[['X3', 'X13', 'X25'], '方向'] == '负').all())

    def test_direction_and_boundaries(self):
        """检验式(5)、负向顺序、端点、越界和模糊对照。"""
        bounds = np.column_stack((np.arange(5), np.arange(1, 6)))
        positive, _ = correlations(0.5, bounds, '正')
        negative, _ = correlations(0.5, bounds, '负')
        np.testing.assert_allclose(positive[:2], [0.5, -0.5])
        np.testing.assert_allclose(negative, positive[::-1])
        boundary, _ = correlations(1, bounds, '正')
        self.assertEqual(boundary.argmax(), 0)
        edge, clipped = correlations(8, bounds, '正')
        self.assertTrue(clipped)
        self.assertEqual(edge.argmax(), 4)
        self.assertAlmostEqual(membership(1.2, bounds).sum(), 1)

    def test_scenarios(self):
        """逐年独立复算，不把实测的 X25 一并剔除。"""
        scores = indicator_scores(*load_data())
        self.assertEqual(len(scores), 125)
        self.assertFalse(scores.duplicated(['编号', '年份']).any())
        assumed = scores[scores.is_assumed]
        self.assertEqual(set(zip(assumed['编号'], assumed['年份'])), {('X25', str(year)) for year in range(2021, 2025)})
        self.assertTrue(np.isfinite(scores[GRADES]).all().all())
        self.assertTrue(((scores[GRADES] >= -1) & (scores[GRADES] <= 0.500000000001)).all().all())
        annual = summarize(scores)
        self.assertEqual(len(annual), 20)
        for _, result in annual.iterrows():
            rows = scores[scores['年份'] == result['年份']]
            if result['阈值口径'] == '仅原经典域':
                rows = rows[~rows['编号'].isin(['X3', 'X13'])]
            if result['情景'] == '剔除假设值':
                rows = rows[~rows.is_assumed]
            self.assertEqual(result['参与项数'], len(rows))
            expected = rows[GRADES].mul(rows['原权重'], axis=0).sum() / rows['原权重'].sum()
            np.testing.assert_allclose(result[GRADES].astype(float), expected)
            self.assertEqual(result['判定等级'], expected.idxmax())
