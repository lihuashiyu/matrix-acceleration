# 00. 设置编译类型参数
set(COMPILE_TYPE 0)                                                            # 1：静态库，2：动态库，其它：可执行文件

# 01. 指定 cmake 最小版本
cmake_minimum_required(VERSION 3.17)

# 02. 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 03. 指定项目的名称，一般和项目的文件夹名称对应
project(matrix-acceleration CUDA)

# 04. 源码文件存放目录
#aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)
aux_source_directory("${PROJECT_SOURCE_DIR}/src" SRC_LIST)

# 05. 设置编译类型
if("${COMPILE_TYPE}" STREQUAL "1")
    add_library(matrix-acceleration  STATIC  ${SRC_LIST})                      # 编译为静态库
elseif("${COMPILE_TYPE}" STREQUAL "2")
    add_library(matrix-acceleration  SHARED  ${SRC_LIST})                      # 编译为动态链接库
else()
    add_executable(matrix-acceleration ${SRC_LIST})                            # 编译为可执行程序
endif()

# 06. 编译输出路径的设置
set(CMAKE_DEBUG_POSTFIX   "-debug")
set(CMAKE_RELEASE_POSTFIX "-release")
set(EXECUTABLE_OUTPUT_PATH          ${PROJECT_SOURCE_DIR}/out)                  # 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)              # 静态库 lib 输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/dll)              # 动态库 dll 输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/exe)              # 可执行 exe 输出路径

# 07. 引用的头文件目录
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
include_directories($ENV{INCLUDE_PATH})

# 08. 引用的库文件目录及文件
set(CUDA_LIB "cublas.lib"  "cuda.lib" "cudadevrt.lib" "cudart_static.lib" "cudadevrt.lib" "cudart_static.lib")
link_directories(${PROJECT_SOURCE_DIR}/lib)                                    # 第三方 lib 库文件搜索路径
target_link_libraries(matrix-acceleration ${CUDA_LIB})                         # 引入库文件到子工程

# 09. 设置属性
set_target_properties(matrix-acceleration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 10. 子项目
# add_subdirectory(vector)
# add_subdirectory(matrix)
# add_subdirectory(digital)
# add_subdirectory(distance)

# 11. 子项目配置
