#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/8 16:35
# @Author : 桐
# @QQ:1041264242
# 注意事项：
self_example_path=r"C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\model\self_model"

def read_python_file(file_path):
    """读取指定路径的Python文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        return code
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None


def select_model_code(main_code,type,data):
    if type == "classification":
        main_code=main_code.replace(data,fr"/mnt/zfy/astro/AutoAstro/Data/ts_data/{data}")
        example_code = read_python_file(
             fr"{self_example_path}\class_model.py")
        return example_code+main_code

    elif type == "prediction":
        example_code = read_python_file(
             fr"{self_example_path}\pre_model.py")
        main_code=main_code.replace(data, fr"/mnt/zfy/astro/AutoAstro/Data/ts_data/{data}")
        return example_code+main_code

    else:
        print("erro")