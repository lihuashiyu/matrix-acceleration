#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
====================================================================================================
    ProjectName   ：  matrix-acceleration
    FileName      ：  test
    CreateTime    ：  2024/5/25 15:38:26
    Author        ：  Lihua Shiyu
    Email         ：  lihuashiyu@github.com
    PythonCompiler：  3.12.10
    Description   ：  matrix 测试
====================================================================================================
"""

from sys import path

path.append("..")
import numpy as np
from api.matrix import load_dll
from api.matrix import vector_add


def vector_add_test(dll):
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    c = vector_add(a, b, dll)
    print(c)


def matrx_add_test(dll):
    pass


def matrix_dot_mul_test(dll):
    pass


def matrix_star_mul_test(dll):
    pass


if __name__ == '__main__':
    dll = load_dll()
    vector_add_test(dll)
