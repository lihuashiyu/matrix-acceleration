#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
====================================================================================================
    ProjectName   ：  matrix-acceleration   
    FileName      ：  matrix    
    CreateTime    ：  2025-07-04 16:40:04 
    Author        ：  lihuashiyu 
    Email         ：  lihuashiyu@github.com 
    PythonCompiler：  3.9.13 
    IDE           ：  CLion 2020.3.4 
    Version       ：  1.0 
    Description   ：  调用 CUDA 代码编译后的 dll，提供 Python 调用接口
====================================================================================================
"""

import os
import sys
import ctypes as cy
import numpy as np
from pathlib import Path


DLL_NAME="matrix-acceleration.dll"                                             # 默认 dll 文件名

def load_dll(path: str = None) -> cy.CDLL:
    """
        加载指定路径的 DLL 文件，若未提供路径则尝试从项目根目录加载。
        
        参数:
            path (str, optional): DLL 文件的路径，默认为 None
        
        返回:
            cy.CDLL: 加载成功的 DLL 对象
        
        异常:
            FileNotFoundError: 若指定路径或默认路径下未找到 DLL 文件
    """
    
    if not path:
        file_path = sys.path[0]                                                # 获取当前文件路径
        project_path = Path(file_path).resolve()                               # 获取项目绝对路径
        path = f"{project_path}/{DLL_NAME}"                                    # 获取 dll 文件路径
    
    # 判断 dll 文件是否存在
    path = path.strip()
    print(path)
    if os.path.exists(path):
        cuda_dll = cy.CDLL(path, winmode=0)                                    # 加载 dll 文件
    else:
        raise FileNotFoundError(f"{path} not found ...")                       # 未找到文件，抛出异常
    
    return cuda_dll


# def vector_add(left_vector: np.ndarray, right_vector: np.ndarray, cuda_dll: cy.CDLL) -> np.ndarray:
#     cuda_dll.vector_add.argtypes = [cy.POINTER(cy.c_int), cy.POINTER(cy.c_int), cy.POINTER(cy.c_int), cy.c_int]
#     cuda_dll.vector_add.restype = cy.POINTER(cy.c_int)
#
#     length = left_vector.shape[0]
#     # left_vector = left_vector.astype(cy.c_int)
#     # right_vector = right_vector.astype(cy.c_int)
#     result_vector = np.zeros(length)
#
#     left_vector_c = (cy.c_int * length)(*left_vector)
#     right_vector_c = (cy.c_int * length)(*right_vector)
#     result_vector_c = (cy.c_int * length)(*result_vector)
#
#     cuda_dll.vector_add(left_vector_c, right_vector_c, result_vector_c, length)
#
#     return result_vector_c
#
# import numpy as np
# import ctypes as cy

def vector_add(left_vector: np.ndarray, right_vector: np.ndarray, cuda_dll: cy.CDLL) -> np.ndarray:
    """
        使用 CUDA DLL 实现两个向量的加法
    
        参数:
            left_vector(np.ndarray)  : 左侧向量
            right_vector(np.ndarray) : 右侧向量
            cuda_dll(CDLL)           : 包含向量加法实现的 CDLL 对象
            
        返回:
            结果向量(np.ndarray)
    """
    
    # 设置一次即可，无需每次调用都设置
    if not hasattr(cuda_dll.vector_add, '_argset'):
        # 配置 C 函数的参数类型和返回值类型
        cuda_dll.vector_add.argtypes = [cy.POINTER(cy.c_int), cy.POINTER(cy.c_int), cy.POINTER(cy.c_int), cy.c_int]
        cuda_dll.vector_add.restype = cy.POINTER(cy.c_int)
        cuda_dll.vector_add._argset = True
        
    length = left_vector.shape[0]                                              # 获取向量的长度

    # 将 NumPy 数组的数据地址转换为 C 语言的指针
    left_c = cy.cast(left_vector.ctypes.data, cy.POINTER(cy.c_int))
    right_c = cy.cast(right_vector.ctypes.data, cy.POINTER(cy.c_int))

    # 初始化结果向量，并将其数据地址转换为 C 语言的指针
    result_vector = np.zeros(length, dtype=np.int32)
    result_c = cy.cast(result_vector.ctypes.data, cy.POINTER(cy.c_int))

    cuda_dll.vector_add(left_c, right_c, result_c, length)                     # 调用 C 函数执行向量加法

    # 返回结果向量
    return result_vector
