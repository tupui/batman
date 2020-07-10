from batman.functions.utils import multi_eval
import numpy.testing as npt


class MultiEval:
    def __init__(self):
        pass

    @multi_eval
    def scalar_func(self, x):
        return x[0] + 2

    @multi_eval
    def vector_func(self, x):
        return [x[0] + 1, x[0] + 2]


def test_scalar():
    func = MultiEval()

    out = func.scalar_func([[1]])
    npt.assert_equal(out, [[3]])

    out = func.scalar_func([[1], [2], [3]])
    npt.assert_equal(out, [[3], [4], [5]])


def test_vector():
    func = MultiEval()

    out = func.vector_func([[1]])
    npt.assert_equal(out, [[2, 3]])

    out = func.vector_func([[1], [2], [3]])
    npt.assert_equal(out, [[2, 3], [3, 4], [4, 5]])
