#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/16 19:55
# @Author : 桐
# @QQ:1041264242
# 注意事项：
from urllib.request import urlopen
import json

url = "http://cdsxmatch.cds.unistra.fr/xmatch/api/v1/sync/tables?action=getVizieRTableNames&select=small&RESPONSEFORMAT=json"

try:
    # 发送请求并读取内容
    with urlopen(url) as response:
        data = json.loads(response.read().decode('utf-8'))

    # 保存到本地
    with open('../data/cross_data/small_tables.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("JSON 内容已保存到 xmatch_tables.json")

except Exception as e:
    print(f"发生错误: {e}")