/**
 ***************************************************************************************************
 * ProjectName   ：  matrix-acceleration 
 * FileName      ：  vector
 * CreateTime    ：  2025-07-03 15:11:05
 * Author        ：  lihuashiyu
 * Email         ：  lihuashiyu@qq.com
 * IDE           ：  CLion 2020.3.4 
 * Version       ：  1.0  
 * Description   ：  向量计算
 ***************************************************************************************************
 */

#pragma once
#include "common.cuh"


#ifndef VECTOR_CUH
    #define VECTOR_CUH

    IMPORT_DLL void vector_add(const float *left_vector, const float *right_vector, float *result_vector, const int length);
#endif
