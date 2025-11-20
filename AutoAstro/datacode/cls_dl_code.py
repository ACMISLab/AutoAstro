#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/12 9:57
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import json
import re

# 正则表达式匹配每一行数据
pattern = re.compile(
    r'\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\[config\]\(([^)]+)\)\s*\|\s*\[model\]\(([^)]+)\)\s*\|'
)

title_pattern = re.compile('# (.*)\n')

with open(r'C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\data\config.md', 'r', encoding='utf-8') as file:

    results = []
    for line in file:
        # print(line.strip())  # 处理每一行内容
        # for line in line.split('\n'):
        title_match = title_pattern.search(line)
        match = pattern.search(line)
        # # print(line)
        # print(match)
        if title_match:
            print(title_match.group(1))
            {
                f"{title_match}": []
            }
        if match:
            print('ok')
            model_data = {
                "Model": match.group(1),
                "Pretrain": match.group(2).strip(),
                "Params_M": float(match.group(3)),
                "Flops_G": float(match.group(4)),
                "Top1_Percent": float(match.group(5)),
                "Top5_Percent": float(match.group(6)),
                "Config": match.group(7),
                "Download": match.group(8)
            }
            results.append(model_data)
        # break

# 转换为JSON格式
json_output = json.dumps(results, indent=2)
print(json_output)

# 将字符串保存到文件（注意这不是有效的JSON格式）
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_output)