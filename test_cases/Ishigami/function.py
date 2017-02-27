# coding:utf-8
from jpod.functions import Ishigami

f_ishigami = Ishigami()

def f(x):

    X1, X2, X3 = x

    return f_ishigami([X1, X2, X3])
