#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/10 12:05
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import json
import os
import subprocess
import zipfile


def up_data_in_json(file_path,new_data):
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，读取现有数据
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                # 如果文件内容不是有效的 JSON，初始化一个空列表
                existing_data = []
    else:
        # 如果文件不存在，初始化一个空列表
        existing_data = []

    # 确保 existing_data 是一个列表
    if not isinstance(existing_data, list):
        existing_data = [existing_data]

    # 追加新数据
    existing_data.append(new_data)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

def get_real_data_path(query_data):
    if query_data["type"] == "local":
        # 如果是本地类型，直接返回本地路径
        print(f"\033[92m Info | 数据集本地路径为: {query_data['load_path']}。\033[0m")
        return query_data
    elif query_data["type"] == "cloud":

        target_path = r"C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\data\image_data"
        git_path= r"C:\Users\10412\.conda\envs\AutoAstro\Library\bin\git.exe"

        # 检查目标路径是否存在
        if os.path.exists(f"{target_path}/{query_data['data_name']}"):
            query_data['down_load_path'] = fr"{target_path}/{query_data['data_name']}"
            print(f"\033[92m Info | 数据集下载成功，保存路径为: {target_path}/{query_data['data_name']}。\033[0m")
            return query_data

        # 执行 git clone 命令
        try:
            subprocess.run([git_path, "clone", f"https://oauth2:TsipEXbLQj99NyJ3-W9P@www.modelscope.cn/datasets/{query_data['load_path']}.git", \
                            fr"{target_path}/{query_data['data_name']}"], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e}")

        # 检查下载路径是否存在
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Downloaded path {target_path} does not exist.")

        # 解压 .zip 文件
        with zipfile.ZipFile(fr"{target_path}/{query_data['data_name']}/{query_data['data_name']}.zip", 'r') as zip_ref:
            zip_ref.extractall(target_path)

        query_data['down_load_path'] = fr"{target_path}/{query_data['data_name']}"

        print(f"\033[92m Info | 数据集下载成功，保存路径为: {target_path}/{query_data['data_name']}。\033[0m")

        # 返回下载内容的路径
        return query_data

    else:
        raise ValueError("Invalid type specified in user_data")


if __name__ == "__main__":

    ##### ---数据录入代码---

    # index=22
    # data_background = """This is RadioAstroVQA's weights. The paper will be soon. Github can be seen in : https://github.com/ACMISLab/RadioAstroVQA"""
    # data_describe = """"""
    #
    # # 示例使用
    # user_data = {
    #     "index": index,
    #     "data_name": "Loamst_Pre",
    #     "task_background":data_background,
    #     "task_describe":data_describe,
    #     "type": "local" , #local or cloud
    #     "load_path": r"C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\data\ts_data\Loamst_Pre.csv"
    # }
    #
    # file_path = r"C:/Users/10412/Desktop/多模态大语言模型/Code/天文Code/AutoAstro/data/data_core.json" #通常保持不动
    #
    # up_data_in_json(file_path=file_path,new_data=user_data)

    # ## ---数据加载代码---
    # query_data =  {
    #     "index": 12,
    #     "data_name": "RadioGalaxyDataset",
    #     "task_background": "Traditional classification tasks",
    #     "task_describe": "The dataset comprises image data designed for morphological classification of radio galaxies, structured into training and test sets. Each image has a resolution of 300×300 pixels and belongs to one of four classes indexed as follows:0:'Bent',1:'Compact',2:'FRI',3:'FRII'.The sample distribution is as follows:Training set—Bent:248,Compact:291,FRI:5,395,FRII:824;Test set—Bent:100,Compact:100,FRI:100,FRII:100.This dataset supports the training and evaluation of classification models for morphological classification of radio galaxies analysis.Data sources:https://zenodo.org/records/7692494",
    #     "type": "cloud",
    #     "load_path": "tongzi/RadioGalaxyDataset"
    # }
    # get_real_data_path(query_data=query_data)

    pass