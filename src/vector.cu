/**
 ***************************************************************************************************
 * ProjectName   ：  matrix-acceleration 
 * FileName      ：  vector
 * CreateTime    ：  2025-07-03 15:11:05
 * Author        ：  lihuashiyu
 * Email         ：  lihuashiyu@qq.com
 * IDE           ：  CLion 2020.3.4 
 * Version       ：  1.0  
 * Description   ：  向量计算头文件
 ***************************************************************************************************
 */

#include "vector.cuh"


/**
 * 向量加法核函数： 对两个输入向量执行逐元素加法操作
 *
 * @param left_vector      左操作数向量，常量指针，输入数据
 * @param right_vector     右操作数向量，常量指针，输入数据
 * @param result_vector    结果向量，输出数据
 * @param length           向量长度，指定向量中元素的总数
 */
GLOBAL_KERNEL vector_add_kernel(const float *left_vector, const float *right_vector, float *result_vector, const int length)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length)
    {
        result_vector[i] = left_vector[i] + right_vector[i];
    }
}



/**
 * 批量向量加法核函数： 对多个输入向量执行逐元素加法操作
 *
 * @param left_vector      左操作数向量，常量指针，输入数据
 * @param right_vector     右操作数向量，常量指针，输入数据
 * @param result_vector    结果向量，输出数据
 * @param length           向量长度，指定向量中元素的总数
 * @param batch_size       批次数量，指定向量批次的数量
 */
GLOBAL_KERNEL vector_add_batch_kernel(const float **left_vector, const float **right_vector, float **result_vector, const int length, const int batch_size)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int bid = blockIdx.y;                                                      // 添加批次维度

    if (i < length && bid < batch_size)
    {
        result_vector[bid][i] = left_vector[bid][i] + right_vector[bid][i];
    }
}



/**
 * 向量加法接口函数（GPU加速）：执行两个单精度浮点向量的逐元素加法运算，使用 CUDA 进行并行计算
 *
 * @param left_vector      指向主机内存中左操作数向量的常量指针
 * @param right_vector     指向主机内存中右操作数向量的常量指针
 * @param result_vector    指向主机内存中存储结果的输出向量指针
 * @param length           向量中元素的总数（正整数）
 */
IMPORT_DLL void vector_add(const float *left_vector, const float *right_vector, float *result_vector, const int length)
{
    // 设备指针声明
    float *left_vector_device, *right_vector_device, *result_vector_device;
    const unsigned int bytes = length * sizeof(float);

    // 分配设备内存
    cudaMallocManaged(&left_vector_device, bytes);
    cudaMallocManaged(&right_vector_device, bytes);
    cudaMallocManaged(&result_vector_device, bytes);

    // 复制数据到设备
    cudaMemcpy(left_vector_device, left_vector, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(right_vector_device, right_vector, bytes, cudaMemcpyHostToDevice);

    // 配置 CUDA 执行参数
    constexpr int block_size = 256;                                            // 每个线程块的线程数
    const int grid_size = (length + block_size - 1) / block_size;              // 向上取整

    // 启动核函数
    vector_add_kernel<<<grid_size, block_size>>>(left_vector_device, right_vector_device, result_vector_device, length);
    // CUDA_CHECK(cudaGetLastError());                                            // 检查核函数启动错误

    // 同步设备确保计算完成
    cudaDeviceSynchronize();

    // 将计算结果从设备内存复制回主机内存
    cudaMemcpy(result_vector, result_vector_device, bytes, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(left_vector_device);
    cudaFree(right_vector_device);
    cudaFree(result_vector_device);

    // printf("\nVector addition is completed by gpu ...... \n");
}
