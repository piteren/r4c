import numpy as np
import unittest

from r4c.helpers import R4Cexception, zscore_norm, da_returns, split_rewards


class TestHelpers(unittest.TestCase):

    def test_zscore_norm(self):
        vals = [1,2,3,4,5]
        norm = zscore_norm(vals)
        print(norm)
        self.assertTrue(type(norm) is np.ndarray)
        self.assertTrue(sum(norm) < 0.001, len(norm) == 5)

    def test_da_returns(self):
        vals = [1,2,3,4,5]
        dar = da_returns(vals, discount=0.5)
        print(dar)
        self.assertTrue(dar == [3.5625, 5.125, 6.25, 6.5, 5.0])

    def test_split_rewards(self):
        vals = [1,2,3,4,5]
        self.assertRaises(R4Cexception, split_rewards, vals, [False,True])
        sr = split_rewards(vals, terminals=[False, True, False, False, True])
        print(sr)
        self.assertTrue(sr == [[1, 2], [3, 4, 5]])
