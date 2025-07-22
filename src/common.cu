/**
 ***************************************************************************************************
 * ProjectName   ：  matrix-acceleration 
 * FileName      ：  common 
 * CreateTime    ：  2025-07-03 16:55:43 
 * Author        ：  lihuashiyu
 * Email         ：  lihuashiyu@github.com
 * IDE           ：  CLion 2020.3.4 
 * Version       ：  1.0  
 * Description   ：  通用函数
 ***************************************************************************************************
 */


#include "common.cuh"


/**
 * 检查 CUDA 运行时错误并进行断言处理
 *
 * @param code  ： cudaError_t 类型，CUDA API 调用返回的错误码
 * @param file  ： const char* 类型，发生错误的文件名（通常通过 __FILE__ 宏传入）
 * @param line  ： int 类型，发生错误的代码行号（通常通过 __LINE__ 宏传入）
 * @param abort ： bool 类型，是否在发生错误时终止程序，默认为 true
 */
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        const char* error_str = cudaGetErrorString(code);                      // 获取可读的错误描述

        // 输出格式化错误信息
        fprintf(stderr, "\n  CUDA 错误：\n");
        fprintf(stderr, "  Code：    %d\n", code);
        fprintf(stderr, "  File：    %s\n", file);
        fprintf(stderr, "  Line：    %d\n", line);
        fprintf(stderr, "  Error：   %s\n", error_str);

        if (abort)                                                             // 终止程序并返回错误码
        {
          exit(code);
        }
    }
}
