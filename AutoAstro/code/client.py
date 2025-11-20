import json

import requests
import time

def submit_code_selfmodel(server_url, python_code, save_path):
    """提交Python代码到服务端执行并获取结果"""
    try:
        # 提交执行请求
        response = requests.post(
            f"{server_url}/execute/selfmodel",
            data={'code': python_code},
            timeout=30
        )

        if response.status_code == 202:
            # 获取任务ID
            task_id = response.json().get('task_id')
            print(f"任务已提交，任务ID: {task_id}")

            # 轮询状态
            while True:
                status_response = requests.get(
                    f"{server_url}/status/{task_id}",
                    timeout=30
                )

                if status_response.status_code == 200:
                    # 检查content-type判断是JSON还是ZIP文件
                    content_type = status_response.headers.get('content-type', '')

                    if 'application/json' in content_type:
                        # JSON响应，处理状态信息
                        status = status_response.json()

                        if status['status'] == 'completed':
                            # 不应该到达这里，completed状态应该返回ZIP文件
                            print("\n执行完成，但未收到结果文件")
                            break
                        elif status['status'] == 'error':
                            print(f"\n错误: {status.get('error', 'Unknown error')}")
                            break
                        else:
                            # 显示进度
                            progress = status.get('progress', 0)
                            print(f"\r执行中... {progress}%", end='', flush=True)
                            time.sleep(5)
                    elif 'application/zip' in content_type:
                        # ZIP文件响应，保存结果
                        with open(f'{save_path}/results.zip', 'wb') as f:
                            f.write(status_response.content)
                        print(f"\n执行完成，结果已保存到 {save_path}/results.zip")
                        return f'{save_path}/results.zip'
                    else:
                        print("\n未知响应类型:", content_type)
                        break
                else:
                    try:
                        error_msg = status_response.json().get('error', 'Unknown error')
                        print(f"\n状态检查失败: {error_msg}")
                    except ValueError:
                        print(f"\n状态检查失败: 无效响应 ({status_response.status_code})")
                    break
        else:
            print(f"错误: {response.json().get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")

def submit_code_mmpre(server_url, python_code, save_path):
    """提交Python代码到服务端执行并获取结果"""
    try:
        # 提交执行请求
        response = requests.post(
            f"{server_url}/execute/mmcls",
            data={'configuration': python_code},
            timeout=30,
            proxies={}  # 禁用代理
        )

        print(response.status_code)

        if response.status_code == 202:
            # 获取任务ID
            task_id = response.json().get('task_id')
            print(f"任务已提交，任务ID: {task_id}")

            # 轮询状态
            while True:
                status_response = requests.get(
                    f"{server_url}/status/{task_id}",
                    timeout=30,
                    proxies = {}  # 禁用代理
                )
                print(status_response)

                if status_response.status_code == 200:
                    # 检查content-type判断是JSON还是ZIP文件
                    content_type = status_response.headers.get('content-type', '')

                    if 'application/json' in content_type:
                        # JSON响应，处理状态信息
                        status = status_response.json()

                        if status['status'] == 'completed':
                            # 不应该到达这里，completed状态应该返回ZIP文件
                            print("\n执行完成，但未收到结果文件")
                            break
                        elif status['status'] == 'error':
                            print(f"\n错误: {status.get('error', 'Unknown error')}")
                            break
                        else:
                            # 显示进度
                            progress = status.get('progress', 0)
                            print(f"\r执行中... {progress}%", end='', flush=True)
                            time.sleep(5)
                    elif 'application/zip' in content_type:
                        # ZIP文件响应，保存结果
                        with open(f'{save_path}/results.zip', 'wb') as f:
                            f.write(status_response.content)
                        print(f"\n执行完成，结果已保存到 {save_path}/results.zip")
                        return f'{save_path}/results.zip'
                    else:
                        print("\n未知响应类型:", content_type)
                        break
                else:
                    try:
                        error_msg = status_response.json().get('error', 'Unknown error')
                        print(f"\n状态检查失败: {error_msg}")
                    except ValueError:
                        print(f"\n状态检查失败: 无效响应 ({status_response.status_code})")
                    break
        else:
            print(f"错误: {response.json().get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")

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


# # 使用示例
# if __name__ == '__main__':
#     server_url = "http://210.40.16.205:30369"  # 替换为实际服务端地址
#     python_code = read_python_file(r"C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\model\self_model\test_pre.py")
#     # python_code = read_python_file(r"C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\model\self_model\prediction.py")
#     result=submit_code_selfmodel(server_url, python_code, save_path=r"C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\data\task_result")

# test = {
#   "framework": {
#     "template": "vgg",
#     "config": "vgg11_8xb32_in1k.py",
#     "pre_weights": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth"
#   },
#   "data_name": "RadioGalaxyDataset",
#   "load_path": "tongzi/RadioGalaxyDataset",
#   "num_classes": 4,
#   "classes": ["Bent", "Compact", "FRI", "FRII"]
# }
#
#
# submit_code_mmpre(server_url, json.dumps(test), save_path=r"C:\Users\10412\Desktop\多模态大语言模型\Code\天文Code\AutoAstro\data\task_result")

# status_response = requests.get(
#     f"{server_url}/status/9026318855504880131",
#     timeout=30,
#     proxies={}  # 禁用代理
# )
#
# print(status_response)