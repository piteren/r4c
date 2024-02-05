import numpy as np
import unittest

from experience_memory import ExperienceMemory

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