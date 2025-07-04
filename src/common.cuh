/**
 ***************************************************************************************************
 * ProjectName   ：  matrix-acceleration 
 * FileName      ：  common 
 * CreateTime    ：  2025-07-03 15:11:05 
 * Author        ：  lihuashiyu
 * Email         ：  lihuashiyu@qq.com
 * IDE           ：  CLion 2020.3.4 
 * Version       ：  1.0  
 * Description   ：  通用函数头文件
 ***************************************************************************************************
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"


#ifndef COMMON_CUH
    #define COMMON_CUH

    // 导出动态库
    #if BUILDING_DLL
        #define IMPORT_DLL extern "C" __declspec(dllimport)
    #else
        #define IMPORT_DLL extern "C" __declspec(dllexport)
    #endif


    // 通用核函数返回类型宏
    #define GLOBAL_KERNEL __global__ void

    // 设备函数返回类型宏
    #define DEVICE_KERNEL __device__ void

    // 主机函数宏（支持CUDA Host 编译）
    #define HOST_KERNEL __host__

    // 检查 CUDA 运行结果
    #define CUDA_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

    // 检查 CUDA 运行时错误
    inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true);
#endif
