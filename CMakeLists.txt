# 00. 设置编译类型参数
set(EXECUTE_FILE  0)                                                           # 可执行文件
set(STATIC_LIB    1)                                                           # 静态库
set(DYNAMIC_LIB   2)                                                           # 动态库
set(COMPILE_TYPE  ${DYNAMIC_LIB})                                              # 编译类型
set(CUDA_VISION   75)                                                          # 70：Volta；75：Ampere；86：Hopper；89：Ada:12：Blackwell

# 01. 指定 cmake 最小版本
cmake_minimum_required(VERSION 3.20)

# 02. 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_VISION}")                                 # GPU 架构编号

# 03. 项目的名称，显式启用 CUDA 支持，一般和项目的文件夹名称对应
project(matrix-acceleration LANGUAGES CXX CUDA)

# 04. 源码文件存放目录
file(GLOB_RECURSE  source_list    "src/*.cu")                                  # 获取所有的头文件
file(GLOB_RECURSE  header_list    "src/*.cuh")                                 # 获取所有的源文件
set(src_list       ${source_list} ${header_list})                              # 拼接变量

# 05. 设置编译类型
if("${COMPILE_TYPE}"     STREQUAL "${STATIC_LIB}")
    add_library(${PROJECT_NAME}       STATIC  ${src_list})                     # 编译为静态库
elseif("${COMPILE_TYPE}" STREQUAL "${DYNAMIC_LIB}")
    add_library("${PROJECT_NAME}"     SHARED  ${src_list})                     # 编译为动态链接库
elseif("${COMPILE_TYPE}" STREQUAL "${EXECUTE_FILE}")
    add_executable("${PROJECT_NAME}" "src/main.cu" "${src_list}")              # 编译为可执行程序
else()
    message(WARNING  "编译类型错误 ...... ")                                    # 编译类型错误
endif()

# 06. 引用的头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 07. 设置构建类型相关的库文件后缀名和输出路径的设置
set(CMAKE_DEBUG_POSTFIX    "-debug")                                           # debug   构建类型的库文件后缀
set(CMAKE_RELEASE_POSTFIX  "-release")                                         # release 构建类型的库文件后缀
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)              # 静态库（.lib 或 .a）输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/dll)              # 动态库（.dll 或 .so）输出目录
set(CMAKE_PDB_OUTPUT_DIRECTORY     ${PROJECT_SOURCE_DIR}/out/pdb)              # 调试符号文件（.pdb）输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/exe)              # 可执行文件（.exe 或 无扩展名）输出目录

# 08. 引用的库文件目录及文件
target_link_libraries("${PROJECT_NAME}"   ${CUDA_LIBRARIES})

# 09. 启用 CUDA 的分离编译功能，允许在多个编译单元中使用 CUDA 代码
set_target_properties("${PROJECT_NAME}"  PROPERTIES  CUDA_SEPARABLE_COMPILATION  ON)

# 10. 添加单元测试的单个可执行文件
file(GLOB_RECURSE  path_list "test/*/*.cu")                                    # 获取所有测试文件
foreach(path  ${path_list})
    string(REPLACE "/" ";" dir_list ${path})                                   # 按照 / 进行分割
    list(GET dir_list -1 file_name)                                            # 获取文件名称
    string(REPLACE "." ";" cache_list ${file_name})                            # 按照 . 进行分割
    list(GET cache_list 0 execute_name)                                        # 获取文件名前缀
    add_executable(${execute_name} ${path})
    # message("${execute_name}: ${path}")
endforeach()
