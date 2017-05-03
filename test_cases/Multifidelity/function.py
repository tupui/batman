# coding:utf-8
from batman.functions import Forrester

f_e = Forrester('e')
f_c = Forrester('c')

def f(x):
    level, X1 = x
    if level == 0:
        sol = f_e([X1])
    else:
        sol = f_c([X1])

    return sol
