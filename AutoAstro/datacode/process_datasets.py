#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/4 20:17
# @Author : 桐
# @QQ:1041264242
# 注意事项：

'''
#验证SDK token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('key-xxxxxx')

#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('tongzi/SpectrumCls')
#您可按需配置 subset_name、split，参照“快速使用”示例代码
'''

import os

mapping={
    'pulsar_time_phase':1,
    'rfi_time_phase':0,
}
type="train"

def generate_image_label_txt(root_dir, output_txt):
    with open(output_txt, 'w') as f:
        for label, folder_name in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder_name)

            try:
                if os.path.isdir(folder_path):
                    for image_name in sorted(os.listdir(folder_path)):
                        # image_path = os.path.join(folder_name, image_name)
                        f.write(f"{type}_subints/{folder_name}/{image_name} {mapping[folder_name]}\n")
            except:
                pass

def data_upload_Ms():
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login('key-xxxxxx')

    # api.upload_folder(
    #     repo_id=f'tongzi/HTRU_Medlat',
    #     folder_path=r'C:\Users\10412\Desktop\test\results\知网\HTRU_Medlat.zip',
    #     commit_message='upload Radio Galaxy dataset folder to repo',
    #     repo_type='dataset'
    # )
    api.upload_folder(
        repo_id=f'tongzi/RadioAstroVQA_Weights',
        folder_path=r'D:\BaiduNetdiskDownload\整理资源\Expand task result file.zip',
        commit_message='upload weights',
        repo_type='dataset'
    )

def open_json():
    import json
    import shutil

    # 打开 JSON 文件
    with open(r'E:\射电天文\扩展版本数据VQA\扩展任务数据集/shuffled_train_data_pingjie_1024_new.json', 'r', encoding='utf-8') as file:
        # 读取文件内容并解析为 Python 对象
        data = json.load(file)

    for sub_data in data:
        flag=False
        # print(sub_data['images'][0].split('/')[-1])
        # print(sub_data['images'][0].split('/')[-2])

        # 定义源文件路径和目标目录路径
        source_file1 = r'E:\射电天文\扩展版本数据VQA\扩展任务数据集\FAST\train\rfi_time_phase/' + sub_data['images'][0].split('/')[-1]  # 替换为你的源文件路径
        source_file2 = r'E:\射电天文\扩展版本数据VQA\扩展任务数据集\FAST\test\rfi_time_phase/' + sub_data['images'][0].split('/')[-1]

        target_directory1 = r'E:\射电天文\扩展版本数据VQA\扩展任务数据集\FAST_NEW\train\rfi_time_phase/'  # 替换为你的目标目录路径
        # target_directory2 = r'E:\射电天文\扩展版本数据VQA\扩展任务数据集\FAST_NEW\test\rfi_best_profile'  # 替换为你的目标目录路径

        if sub_data['images'][0].split('/')[-2]=='pulsar_pingjie':
            # try:
            #     shutil.move(source_file1, target_directory1)
            #     Flag=True
            # except:
            #     try:
            #         shutil.move(source_file2, target_directory1)
            #         Flag = True
            #     except:
            #         Flag = False
            #
            # if flag==False:
            #     print("移动图片有问题")
            pass

        elif sub_data['images'][0].split('/')[-2]=='rfi_pingjie':
            try:
                shutil.move(source_file1, target_directory1)
                flag = True
            except:
                try:
                    shutil.move(source_file2, target_directory1)
                    flag = True
                except:
                    flag = False

            if flag == False:
                print("移动图片有问题")
        else:
            print('erro:未识别标签')
        # break

def ATLAS_MeerLICHT_PANSTARRS_NPY():
    from helper_functions import save_picture
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import json

    # Load the dataset
    file_path_data = r'E:\射电天文\扩展版本数据VQA\扩展任务数据集\Dataset for Large Language Models Classification of Astronomical Transient/ATLAS_images.npy' #
    file_path_labels_csv = r'E:\射电天文\扩展版本数据VQA\扩展任务数据集\Dataset for Large Language Models Classification of Astronomical Transient/ATLAS_labels.csv' #

    triplets = np.load(file_path_data)

    # Load labels from CSV file
    labels_df = pd.read_csv(file_path_labels_csv)

    # 随机拆分为训练集和测试集，比例为 8:2
    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42)

    print(train_df)
    print(test_df)
    # print(labels_df)

    # Load train data directly from the dataset already loaded
    train_data = triplets  # No need to download or reload; reuse the loaded data
    save=r"ATLAS/test" #
    datas=[]

    for data in test_df.itertuples(): #
        names=save_picture(train_data, data.index_no, False, save)
        message={
        "conversations": [
            {"from": "user", "value": f"<img>{save}/{names[0]}</img><img>{save}/{names[1]}</img><img>{save}/{names[2]}</img>Look at the 3 images (New, Reference and Difference images) and classify if the source at the centre of the cutout and inside the red circle is a Real or Bogus astronomical transient. Return the answer in an Assignment Statement, containing ONE variable: IDENTIFICATION. Only return the IDENTIFICATION, not the Description."},
            {"from": "assistant", "value": f"IDENTIFICATION = '{data.label}'"}
            ]
        }
        datas.append(message)

    # 将内容写入 JSON 文件
    with open(r'E:\射电天文\扩展版本数据VQA\扩展任务数据集\Dataset for Large Language Models Classification of Astronomical Transient\ATLAS\meta/test.json', "w") as json_file: #
        json.dump(datas, json_file, indent=4)


    # # Save pictures for sample indexes
    # save_picture(train_data, 0, True,path="")
    #
    # message={
    # "conversations": [
    #     {"from": "user", "value": f"<img>{}</img><img>{}</img><img>{}</img>Look at the 3 images (New, Reference and Difference images) and classify if the source at the centre of the cutout and inside the red circle is a Real or Bogus astronomical transient. Return the answer in an Assignment Statement, containing ONE variable: IDENTIFICATION. Only return the IDENTIFICATION, not the Description."},
    #     {"from": "assistant", "value": f"IDENTIFICATION = '{}'"}
    #     ]
    # }


if __name__ == "__main__":
    # 使用示例
    # root_directory = fr'E:\射电天文\扩展版本数据VQA\扩展任务数据集\FAST_NEW\{type}'  # 替换为你的图像文件夹路径
    # # output_txt_file = f'{type}.txt'  # 输出的文本文件路径
    # output_txt_file = f'{type}_subints.txt'  # 输出的文本文件路径
    #
    # generate_image_label_txt(root_directory, output_txt_file)

    #上传
    data_upload_Ms()

    # open_json()

    # ATLAS_MeerLICHT_PANSTARRS_NPY()
    pass