#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
====================================================================================================
    ProjectName   ：  matrix-acceleration   
    FileName      ：  matrix    
    CreateTime    ：  2025-07-04 16:40:04 
    Author        ：  lihuashiyu 
    Email         ：  lihuashiyu@github.com
    Description   ：  调用 CUDA 代码编译后的 dll，提供 Python 调用接口
====================================================================================================
"""

from os import path
import ctypes as cy
import numpy as np


DLL_NAME="matrix-acceleration.dll"                                             # 默认 dll 文件名

def load_dll(dll_path: str = None) -> cy.CDLL:
    """
        加载指定路径的 DLL 文件，若未提供路径则尝试从项目根目录加载。
        
        参数:
            path (str, optional): DLL 文件的路径，默认为 None
        
        返回:
            cy.CDLL: 加载成功的 DLL 对象
        
        异常:
            FileNotFoundError: 若指定路径或默认路径下未找到 DLL 文件
    """
    
    if not dll_path:
        file_path = path.abspath(__file__)                                     # 获取当前文件路径
        file_dir = path.dirname(file_path)                                     # 获取当前文件目录
        dll_path = f"{file_dir}/{DLL_NAME}"                                    # 获取 dll 文件路径
    else:
        dll_path = dll_path.strip()
        
    # 判断 dll 文件是否存在
    if path.exists(dll_path):
        cuda_dll = cy.CDLL(dll_path, winmode=0)                                # 加载 dll 文件
    else:
        raise FileNotFoundError(f"{dll_path} not found ...")                   # 未找到文件，抛出异常
    
    return cuda_dll


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
