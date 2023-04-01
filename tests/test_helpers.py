import numpy as np
import unittest

from r4c.helpers import RLException, ExperienceMemory, zscore_norm, discounted_return, movavg_return, split_rewards


class TestExperienceMemory(unittest.TestCase):

    def test_base(self):

        em = ExperienceMemory()

        em.add({
            'observations':         [1,2,3],
            'actions':              [1,2,3],
            'rewards':              [1,2,3],
            'next_observations':    [1,2,3],
            'terminals':            [1,2,3],
            'wons':                 [1,2,3]})

        print(len(em))
        self.assertTrue(len(em) == 3)

        exp = em.get_all()
        print(exp)
        self.assertTrue(len(exp) == 6 and len(exp['observations']) == 3 and type(exp['observations']) is np.ndarray)
        self.assertTrue(len(em) == 0)


    def test_exception(self):

        em = ExperienceMemory()

        self.assertRaises(Exception, em.add, {'sth': [1,2,3]})


    def test_zscore_norm(self):

        vals = [1,2,3,4,5]
        norm = zscore_norm(vals)
        print(norm)
        self.assertTrue(sum(norm) < 0.001, len(norm) == 5)


    def test_discounted_return(self):

        vals = [1,2,3,4,5]
        dr = discounted_return(vals, discount=0.5)
        print(dr)
        self.assertTrue(dr == [3.5625, 5.125, 6.25, 6.5, 5.0])


    def test_movavg_return(self):

        vals = [1,2,3,4,5]
        mr = movavg_return(vals, factor=0.5)
        print(mr)
        self.assertTrue(mr == [1.9375, 2.875, 3.75, 4.5, 5.0])


    def test_split_rewards(self):

        vals = [1,2,3,4,5]

        self.assertRaises(RLException, split_rewards, vals, [False,True])

        sr = split_rewards(vals, terminals=[False, True, False, False, True])
        print(sr)
        self.assertTrue(sr == [[1, 2], [3, 4, 5]])
