# _*_ coding:utf-8 _*_
import subprocess
from tqdm import tqdm
import sys
param_list = []
for pseudo_threshold in ["10","20","30","40","50","60","70","80","90","95"]:
    param_list.append({"pseudo_threshold": pseudo_threshold})
with open("testcodecoraml.txt", "w", encoding='UTF-8', errors='ignore') as file:
    for params in tqdm(param_list, desc='Processing parameters'):
        command = ["python", "main3.py", f"--pseudo_threshold={params['pseudo_threshold']}"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate()  # 获取子进程的输出和错误
        # # 将输出写入文件
        file.write(f"{params}\n")
        # file.write(stdout)
        # file.write(stderr)
        # file.write('\n\n')

        print(stdout, end='')  # 显示在控制台
        file.write(stdout)  # 同时写入文件

        print(stderr, end='')  # 显示在控制台
        file.write(stderr)  # 同时写入文件

        file.write('\n\n')

