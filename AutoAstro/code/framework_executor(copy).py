#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/11 14:47
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from llm_api import deepseek_chat
from analyze import visual_analyze_png,data_analyze_md,linking_insight,update_node_content,task_eval,subtask_decompose,insert_history_tree
from error_correction import code_correction,check_for_none,outlier_search
from prompt import vis_prompt,data_prompt,train_prompt,self_model_prompt,image_cls_model_prompt
from client import submit_code_selfmodel,submit_code_mmpre
from AutoAstro.model.self_model.ts_model_example import select_model_code
from tools import parse_execution_order
import re

def mmpretrain_executor(data_intr,fold_path,requirements,pre_execute_task,mm_example,history_json_path):
    print(f"\033[92m Info | 执行Step4: 开始执行任务。\033[0m")

    for index,task in enumerate(pre_execute_task):
        index=index+1

        try:
            print(f"\033[93m    Info | 执行子任务: 构建模型配置\033[0m")

            with open(mm_example, 'r', encoding='utf-8') as file:
                models = json.load(file)

            model_des = models[task['type']]

            image_cls_model_execute_prompt = image_cls_model_prompt(requirements,models,data_intr,task['type'])
            print(image_cls_model_execute_prompt)

            # print(image_cls_model_execute_prompt)
            sm_answer, history = deepseek_chat(user=image_cls_model_execute_prompt, history_message=[], system="")
            print(sm_answer)

            auto_config = extract_json(sm_answer)

            configuration = None
            for select_model in model_des:
                if select_model["Model"] == auto_config["model"]:
                    # print(select_model)
                    configuration = {
                        "framework": {
                            "template": task['type'],
                            "config": select_model["Config"],
                            "pre_weights": select_model["Download"]
                        },
                        "data_name": data_intr['data_name'],
                        "load_path": data_intr['load_path'],
                        "num_classes": auto_config["num_classes"],
                        "classes": auto_config["classes"]
                    }
                    update_node_content(target_id=task['task_id'], new_content={"explanation": auto_config["explanation"]},
                                        history_json=history_json_path)
            print(configuration)

            # 确保目录存在
            os.makedirs(fr"{fold_path}\sub_task{index}", exist_ok=True)  # Create the directory if it doesn't exist
            submit_code_mmpre("http://10.10.10.56:34260", json.dumps(configuration), save_path=fr"{fold_path}\sub_task{index}")
            update_node_content(target_id=task['task_id'], new_content={"done": True}, history_json=history_json_path)
            print(f"\033[92m    Success | 执行子任务任务id:{task['task_id']}, 执行成功\033[0m")

        except Exception as e:
            print(f"发生了异常：{e}")
            print(f"\033[91m    Info | 执行子任务任务id:{task['task_id']}, 执行失败\033[0m")

def machine_learning_executor(data_intr,fold_path,requirements,data_summary,pre_execute_task,history_json_path,save_path):


    print(f"\033[92m Info | 执行Step4: 开始执行任务。\033[0m")

    for index,data in enumerate(pre_execute_task):
        index=index+1
        try:
            execute=data['task']

            print(f"\033[93m    Info | 执行子任务任务{index}:可视化代码生成\033[0m")
            # 可视化分析代码生成
#             vis_execute_prompt=f"""# Instruction
# According to whether the current task needs to perform a visual analysis, make the following decision:
# - If no visual analysis is needed in the task description (e.g., only train the model or perform statistical analysis on the data), return `Over`, and do not return any other values.
# - If visual analysis is needed in the task description, proceed to write Python code to analyze the data and solve this task. After finishing the data analysis, use Altair to generate an interactive visualization. Add a brush function, tooltip, and legend if different colors are used. Ensure that each generated visualization has a brush function that allows you to select a subset of the data. Please ensure that only one view is generated. No combination of views is required. If the dataset is large, use stratified sampling by category to ensure representativeness, keeping the total sample size ≤ 1000 for efficient and clear visualization.
#
#
# # User Input
# - Task description: {execute["task_description"]}
# - Data description: '''{data_intr["task_describe"]}'''
# - Data summary: {data_summary}
# - Task explanation: {execute["explanation"]}
# - Selected columns: {execute["selected_columns"]}
# - Data Volume
#
# # Output Example
# ```python
# import pandas as pd
# import altair as alt
# def execute(data: pd.DataFrame):
#     # Data preprocessing
#          <codes>
#     # Chart generation
#     chart = alt.Chart().mark_bar().encode().properties()
#     return chart
# ```"""

            vis_execute_prompt = vis_prompt(execute["task_description"],data_intr["task_describe"],data_summary,execute["explanation"],execute["selected_columns"])

            print(vis_execute_prompt)
            vis_answer, history = deepseek_chat(user=vis_execute_prompt, history_message=[], system="")
            print(vis_answer)

            if check_for_none(vis_answer):
                print(f"\033[93m    Info | 执行子任务任务{index}:无需进行可视化分析\033[0m")
            else:
                pre_vis_code=extract_code(text=vis_answer)+f"""
df = pd.read_csv(r"{save_path}")
final_chart=execute(df)
# final_chart.save(r'{fold_path}/sub_task{index}.html', embed_options={{'renderer':'svg'}})
# final_chart.save(r'{fold_path}/sub_task{index}.png')
final_chart.savefig(r'{fold_path}/sub_task{index}.png')
plt.close(final_chart)"""
                vis_erro = write_and_execute(code=pre_vis_code, file_path=fr"{fold_path}/sub_task{index}_viscode_executor.py", column=execute["selected_columns"])
                print(vis_erro )


            print(f"\033[93m    Info | 执行子任务任务{index}:数据分析代码生成\033[0m")
            #数据分析代码生成
            data_execute_prompt = data_prompt(execute["task_description"],data_intr["task_describe"],data_summary,execute["explanation"],execute["selected_columns"])

            dat_answer, history = deepseek_chat(user=data_execute_prompt, history_message=[], system="")
            print(dat_answer)
            if check_for_none(dat_answer):
                print(f"\033[93m    Info | 执行子任务任务{index}:无需进行数据分析\033[0m")
            else:
                pre_dat_code = extract_code(text=dat_answer) + f"""
pd.set_option('display.max_rows', 10)  # 显示所有行
pd.set_option('display.max_columns', 10)  # 显示所有列
#pd.set_option('display.width', 1000)  # 设置输出宽度
pd.set_option('display.max_colwidth', None)  # 取消列内容截断
df = pd.read_csv(r"{save_path}")
results=execute(df)
result_shown=""
index=0
for key, value in results.items():
    index+=1
    result_shown+=f"{{index}}. **{{key}}**:\\n{{value}}\\n\\n"
    
with open(r'{fold_path}/sub_code_task{index}.md', "w", encoding="utf-8") as md_file:
    md_file.write(result_shown)"""
                dat_erro = write_and_execute(code=pre_dat_code, file_path=fr"{fold_path}/sub_task{index}_datcode_executor.py", column=execute["selected_columns"])
                print(dat_erro)


            print(f"\033[93m    Info | 执行子任务任务{index}:模型训练代码生成\033[0m")
            # 确保目录存在
            os.makedirs(fr"{fold_path}\img", exist_ok=True)  # Create the directory if it doesn't exist

            #训练模型代码生成
            train_execute_prompt = train_prompt(execute["task_description"],data_intr["task_describe"],data_summary,execute["explanation"],execute["selected_columns"],fold_path)

            train_answer, history = deepseek_chat(user=train_execute_prompt, history_message=[], system="")
            print(train_answer)


            if check_for_none(train_answer):
                print(f"\033[93m    Info | 执行子任务任务{index}:无需进行模型训练\033[0m")
            else:
                pre_tra_code = extract_code(text=train_answer)+f"""
df = pd.read_csv(r"{save_path}")
results=execute(df)
index=0
result_shown=""
for key, value in results.items():
    index+=1
    result_shown+=f"{{index}}. **{{key}}**:\\n{{value}}\\n\\n"
with open(r'{fold_path}/sub_tra_task{index}.md', "w", encoding="utf-8") as md_file:
    md_file.write(result_shown)"""
                # print(pre_tra_code)
                tra_erro = write_and_execute(code=pre_tra_code, file_path=fr"{fold_path}/sub_task{index}_tracode_executor.py", column=execute["selected_columns"])
                print(tra_erro)

            if check_for_none(vis_answer):
                print(f"\033[93m    Info | 执行子任务任务{index}:无需进行可视化分析\033[0m")
            else:
                print(f"\033[93m    Info | 执行子任务任务{index}:可视化分析\033[0m")
                #可视化分析
                visual_analyze_result=visual_analyze_png(execute["task_description"],data_intr["task_describe"],fr'{fold_path}/sub_task{index}.png')


            if check_for_none(dat_answer):
                print(f"\033[93m    Info | 执行子任务任务{index}:无需进行数据分析\033[0m")
            else:
                print(f"\033[93m    Info | 执行子任务任务{index}:数据分析\033[0m")
                #数据分析
                data_analyze_result = data_analyze_md(execute["task_description"], data_intr["task_describe"],fr'{fold_path}/sub_code_task{index}.md')


            print(f"\033[93m    Info | 执行子任务任务{index}:中间过程分析结果写入\033[0m")
            # 假设指定的 Markdown 文件路径为 output_path
            md_output_path = fr"{fold_path}/sub_sum_ori_task{index}.md"
            # 打开或创建 Markdown 文件
            with open(md_output_path, 'a', encoding='utf-8') as md_file:

                if check_for_none(vis_answer)==False:
                    # 写入图像分析结果
                    image_markdown = fr"![sub_task{index}]({fold_path}/sub_task{index}.png)"
                    md_file.write("# visual_analyze_result\n"+image_markdown + "\n\n")
                    md_file.write(visual_analyze_result + "\n\n")

                if check_for_none(dat_answer)==False:
                    # 写入数据分析结果
                    md_file.write("# data_analyze_result\n" + data_analyze_result + "\n\n")

            if check_for_none(vis_answer)==False and check_for_none(dat_answer)==False:
                print(f"\033[93m    Info | 执行子任务任务{index}:子任务结果整合\033[0m")
                visual_insight = "# visual_analyze_result\n" + image_markdown + "\n\n" + visual_analyze_result + "\n\n"
                data_insight = "# data_analyze_result\n" + data_analyze_result + "\n\n"
                answer=linking_insight(execute["task_description"],visual_insight,data_insight)
                with open(fr"{fold_path}/sub_sum_norm_task{index}.md", 'a', encoding='utf-8') as md_file:
                    md_file.write(answer)

            eval_answer = task_eval(requirements=requirements, task=execute["task_description"], insight=answer, column=execute["selected_columns"])
            eval_result = extract_json(text=eval_answer)
            update_node_content(target_id=data['task_id'],new_content={"done":True}, history_json=history_json_path)
            print(f"\033[92m    Success | 执行子任务任务id:{data['task_id']}, 执行成功\033[0m")

            subtask_answer = subtask_decompose(requirements=requirements,task=execute["task_description"],usedcolumn=execute["selected_columns"],data=data_intr["task_describe"],insight=answer,score=eval_result['score'],reasons=eval_result['explanation'])
            subtask_result = extract_json(text=subtask_answer)
            subtask_execution(data['task_id'], subtask_result, data_intr, fold_path, requirements, history_json_path, save_path)

        except Exception as e:
            print(f"发生了异常：{e}")
            print(f"\033[91m    Info | 执行子任务任务id:{data['task_id']}, 执行失败\033[0m")
        # break

def self_model_executor(data_intr,fold_path,requirements,data_summary,pre_execute_task,history_json_path,save_path):

    print(f"\033[92m Info | 执行Step4: 开始执行任务。\033[0m")

    for index,data in enumerate(pre_execute_task):
        index=index+1

        try:
            execute=data['task']

            if execute["task_type"] != "other":
                print(f"\033[93m    Info | 执行子任务任务{index}:时序深度学习模型训练\033[0m")
                self_model_execute_prompt = self_model_prompt(execute["task_description"], data_intr["task_describe"], data_summary, execute["explanation"], execute["selected_columns"], data_intr["data_name"]+".csv", execute["task_type"])
                print(self_model_execute_prompt)

                sm_answer, history = deepseek_chat(user=self_model_execute_prompt, history_message=[], system="")
                print(sm_answer)

                generate_code = extract_code(text=sm_answer)
                execute_code = select_model_code(main_code=generate_code, type=execute["task_type"], data=data_intr["data_name"]+".csv")
                # 确保目录存在
                os.makedirs(fr"{fold_path}\sub_task{index}", exist_ok=True)  # Create the directory if it doesn't exist
                submit_code_selfmodel("http://10.10.10.56:34260", execute_code, save_path=fr"{fold_path}\sub_task{index}")

                update_node_content(target_id=data['task_id'], new_content={"done": True}, history_json=history_json_path)
                print(f"\033[92m    Success | 执行子任务任务id:{data['task_id']}, 执行成功\033[0m")

            else:
                print(f"\033[93m    Info | 执行子任务任务{index}:可视化代码生成\033[0m")
                vis_execute_prompt = vis_prompt(execute["task_description"], data_intr["task_describe"], data_summary, execute["explanation"], execute["selected_columns"])

                print(vis_execute_prompt)
                vis_answer, history = deepseek_chat(user=vis_execute_prompt, history_message=[], system="")
                print(vis_answer)

                if check_for_none(vis_answer):
                    print(f"\033[93m    Info | 执行子任务任务{index}:无需进行可视化分析\033[0m")
                else:
                    pre_vis_code = extract_code(text=vis_answer) + f"""
df = pd.read_csv(r"{save_path}")
final_chart=execute(df)
# final_chart.save(r'{fold_path}/sub_task{index}.html', embed_options={{'renderer':'svg'}})
# final_chart.save(r'{fold_path}/sub_task{index}.png')
final_chart.savefig(r'{fold_path}/sub_task{index}.png')
plt.close(final_chart)"""
                    vis_erro = write_and_execute(code=pre_vis_code, file_path=fr"{fold_path}/sub_task{index}_viscode_executor.py", column=execute["selected_columns"])
                    print(vis_erro)

                print(f"\033[93m    Info | 执行子任务任务{index}:数据分析代码生成\033[0m")
                # 数据分析代码生成
                data_execute_prompt = data_prompt(execute["task_description"], data_intr["task_describe"], data_summary, execute["explanation"], execute["selected_columns"])

                dat_answer, history = deepseek_chat(user=data_execute_prompt, history_message=[], system="")
                print(dat_answer)

                if check_for_none(dat_answer):
                    print(f"\033[93m    Info | 执行子任务任务{index}:无需进行数据分析\033[0m")
                else:
                    pre_dat_code = extract_code(text=dat_answer) + f"""
pd.set_option('display.max_rows', 10)  # 显示所有行
pd.set_option('display.max_columns', 10)  # 显示所有列
#pd.set_option('display.width', 1000)  # 设置输出宽度
pd.set_option('display.max_colwidth', None)  # 取消列内容截断
df = pd.read_csv(r"{save_path}")
results=execute(df)
result_shown=""
index=0
for key, value in results.items():
    index+=1
    result_shown+=f"{{index}}. **{{key}}**:\\n{{value}}\\n\\n"

with open(r'{fold_path}/sub_code_task{index}.md', "w", encoding="utf-8") as md_file:
    md_file.write(result_shown)"""
                    dat_erro = write_and_execute(code=pre_dat_code, file_path=fr"{fold_path}/sub_task{index}_datcode_executor.py", column=execute["selected_columns"])
                    print(dat_erro)

                if check_for_none(vis_answer):
                    print(f"\033[93m    Info | 执行子任务任务{index}:无需进行可视化分析\033[0m")
                else:
                    print(f"\033[93m    Info | 执行子任务任务{index}:可视化分析\033[0m")
                    #可视化分析
                    visual_analyze_result=visual_analyze_png(execute["task_description"],data_intr["task_describe"],fr'{fold_path}/sub_task{index}.png')


                if check_for_none(dat_answer):
                    print(f"\033[93m    Info | 执行子任务任务{index}:无需进行数据分析\033[0m")
                else:
                    print(f"\033[93m    Info | 执行子任务任务{index}:数据分析\033[0m")
                    #数据分析
                    data_analyze_result = data_analyze_md(execute["task_description"], data_intr["task_describe"],fr'{fold_path}/sub_code_task{index}.md')

                print(f"\033[93m    Info | 执行子任务任务{index}:中间过程分析结果写入\033[0m")
                # 假设指定的 Markdown 文件路径为 output_path
                md_output_path = fr"{fold_path}/sub_sum_ori_task{index}.md"
                # 打开或创建 Markdown 文件
                with open(md_output_path, 'a', encoding='utf-8') as md_file:

                    if check_for_none(vis_answer) == False:
                        # 写入图像分析结果
                        image_markdown = fr"![sub_task{index}]({fold_path}/sub_task{index}.png)"
                        md_file.write("# visual_analyze_result\n" + image_markdown + "\n\n")
                        md_file.write(visual_analyze_result + "\n\n")

                    if check_for_none(dat_answer) == False:
                        # 写入数据分析结果
                        md_file.write("# data_analyze_result\n" + data_analyze_result + "\n\n")

                if check_for_none(vis_answer) == False and check_for_none(dat_answer) == False:
                    print(f"\033[93m    Info | 执行子任务任务{index}:子任务结果整合\033[0m")
                    visual_insight = "# visual_analyze_result\n" + image_markdown + "\n\n" + visual_analyze_result + "\n\n"
                    data_insight = "# data_analyze_result\n" + data_analyze_result + "\n\n"
                    answer = linking_insight(execute["task_description"], visual_insight, data_insight)
                    with open(fr"{fold_path}/sub_sum_norm_task{index}.md", 'a', encoding='utf-8') as md_file:
                        md_file.write(answer)

                update_node_content(target_id=data['task_id'], new_content={"done": True}, history_json=history_json_path)
                print(f"\033[92m    Success | 执行子任务任务id:{data['task_id']}, 执行成功\033[0m")

        except Exception as e:
            print(f"发生了异常：{e}")
            print(f"\033[91m    Info | 执行子任务任务id:{data['task_id']}, 执行失败\033[0m")

def subtask_execution(pre_id,subtask_result,data_intr,fold_path,requirements,history_json_path,save_path):

    orders = parse_execution_order(subtask_result['execution order'])

    sub_temp = []
    for index, order in enumerate(orders):
        for sub_index in order:
            # 读取 JSON 文件
            with open(history_json_path, 'r', encoding='utf-8') as file:
                node = json.load(file)

            new_data = {
                "id": node['toatl']+1,
                "task": subtask_result["task"][sub_index],
                "explanation": "",
                "type": subtask_result["task type"][sub_index],
                "done": False,
                "children": [],
            }

            #插入子任务
            if index==0:
                insert_history_tree(target_id=pre_id, new_data=new_data, history_json=history_json_path)
            else:
                for data in sub_temp:
                    insert_history_tree(target_id=data["id"], new_data=new_data, history_json=history_json_path)

            #任务执行
            try:
                print(f"\033[93m    Info | 执行子任务任务{index}:可视化代码生成\033[0m")
            except:
                pass


            insight=""

            if index == 0:
                sub_temp.append({"id":new_data["id"],"insight": insight})





    # print(subtask_result)
    # print("--------------------------------------------------")
    # print(data_intr)
    pass


# 读取CSV文件
def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None

#代码提取
def extract_code(text):
    # 使用正则表达式匹配Python代码块
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        python_code = match.group(1)
        return python_code
    else:
        print("No Python code found.")

#json提取
def extract_json(text):
    # 使用正则表达式匹配Python代码块
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        message_json = match.group(1)
        return json.loads(message_json)
    else:
        print("No json found.")

#代码执行
def write_and_execute(code: str, file_path: str, column: str):
    max_attempts = 3  # 最大尝试修正次数
    attempt = 0

    while attempt < max_attempts:
        try:
            # 写入代码到指定文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # 执行该 Python 文件
            result = subprocess.run([r'C:\Users\10412\.conda\envs\AutoAstro\python.exe', file_path], capture_output=True, text=True, encoding="utf-8", errors="replace")

            # 检查是否有错误
            if result.returncode != 0:
                # 如果执行失败，调用修正函数
                print(f"erro: {result.stderr}")
                code = code_correction(result.stderr, code, column)
                attempt += 1
            else:
                # 执行成功，返回结果
                return f"Execution succeeded: {result.stdout} attempts:{attempt}"
        except Exception as e:
            # 捕获其他异常，调用修正函数
            code = code_correction(str(e), code, column)
            attempt += 1

    # 如果超过最大尝试次数仍然失败，返回报错异常
    return f"Execution failed after {max_attempts} attempts: {result.stderr if 'result' in locals() else str(e)}"


