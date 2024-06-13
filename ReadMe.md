# bigdata-deploy

  使用 cuda 对矩阵进行加速

## 1. 目录结构

```shell
    matrix-acceleration                                                        # 根目录
        ├───api                                                               # python 调用的 api
        ├───doc                                                               # 文档说明目录
        ├───include                                                           # cuda 代码需要的第三方头目录
        ├───lib                                                               # cuda 代码需要的第三方库文目录
        ├───src                                                               # cuda 代码源目睹
        ├───test                                                              # 单元测试代码
        ├───CMakeLists.txt                                                    # cmake 配置文件
        ├───LICENSE                                                           # GPL 协议说明
        └───ReadMe.md                                                         # 自述文件 
```

## 2. 类型
```
    文件命名方式  ： PascalCase，帕斯卡命名
    函数命名方式  ： camelCase， 小写驼峰
    宏定义命名方式： UPPER_SNAKE_CASE，大写下划线
    常量命名方式  ： UPPER_SNAKE_CASE，大写下划线
    变量命名方式  ： lower_snake_case，小写下划线
    结构体命名方式： PascalCaseStruc，帕斯卡命名加Struc后缀
    枚举值命名方式： PascalCaseEnum，帕斯卡命名加Enum后缀
```
