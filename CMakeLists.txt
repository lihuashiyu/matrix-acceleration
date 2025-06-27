# 00. 设置编译类型参数
set(COMPILE_TYPE  2)                                                           # 0：可执行文件；1：静态库；2：动态库，
set(EXECUTE_FILE  0)                                                           # 可执行文件
set(STATIC_LIB    1)                                                           # 静态库
set(DYNAMIC_LIB   2)                                                           # 动态库
set(CUDA_VISION   75)                                                          # 70：Volta；75：Ampere；86：Hopper；89：Ada:12：Blackwell

# 01. 指定 cmake 最小版本
cmake_minimum_required(VERSION 3.20)

# 02. 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_VISION}")                                 # GPU 架构编号

# 03. 指定项目的名称，一般和项目的文件夹名称对应
project(matrix-acceleration CUDA)

# 04. 源码文件存放目录
file(GLOB_RECURSE  source_list    "src/*/*.cu")                                # 获取所有的头文件
file(GLOB_RECURSE  header_list    "src/*/*.cuh")                               # 获取所有的源文件
set(src_list       ${source_list} ${header_list})                              # 拼接变量

# 05. 设置编译类型
if("${COMPILE_TYPE}"     STREQUAL "${STATIC_LIB}")
    add_library(matrix-acceleration  STATIC  ${src_list})                      # 编译为静态库
elseif("${COMPILE_TYPE}" STREQUAL "${DYNAMIC_LIB}")
    add_library(matrix-acceleration  SHARED  ${src_list})                      # 编译为动态链接库
elseif("${COMPILE_TYPE}" STREQUAL "${EXECUTE_FILE}")
    add_executable(matrix-acceleration "src/main.cu" "${src_list}")            # 编译为可执行程序
else()
    message(WARNING  "编译类型错误 ...... ")                                     # 编译类型错误
endif()

# 06. 编译输出路径的设置
set(CMAKE_DEBUG_POSTFIX   "-debug")
set(CMAKE_RELEASE_POSTFIX "-release")
set(EXECUTABLE_OUTPUT_PATH         ${PROJECT_SOURCE_DIR}/out)                  # 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)              # 静态库 lib 输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/dll)              # 动态库 dll 输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/exe)              # 可执行 exe 输出路径

# 07. 引用的头文件目录
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_PATH})

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

# 12. 添加测试单个可执行文件
file(GLOB_RECURSE  path_list "test/cuda/*.cu")                                # 获取所有测试文件
foreach(path ${path_list})
    string(REPLACE "/" ";" dir_list ${path})                                   # 按照 / 进行分割
    list(GET dir_list -1 file_name)                                            # 获取文件名称
    string(REPLACE "." ";" cache_list ${file_name})                            # 按照 . 进行分割
    list(GET cache_list 0 execute_name)                                        # 获取文件名前缀
    add_executable(${execute_name} ${path})
endforeach()
