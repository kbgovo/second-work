# _*_ coding:utf-8 _*_
#batchrun for baseline
import subprocess
from tqdm import tqdm
param_list = []
for edge_rate in ["0.2", "0.3"]:
    for s_rate in ["0.3", "0.4", "0.5"]:
            for dataset in ["dblp"]:
                param_list.append({"edge_rate": edge_rate, "s_rate": s_rate, "dataset": dataset})
with open("test_code_dblp.txt", "w", encoding='UTF-8', errors='ignore') as file:
    for params in tqdm(param_list, desc='Processing parameters'):
        if params['dataset'] == 'cora' or params['dataset'] == 'citeseer':
            command = ["python", "main2.py", f"--edge_rate={params['edge_rate']}", f"--s_rate={params['s_rate']}",
                       f"--dataset={params['dataset']}", "--label_rate=0.05"]
        else:
            command = ["python", "main2.py", f"--edge_rate={params['edge_rate']}", f"--s_rate={params['s_rate']}",
                       f"--dataset={params['dataset']}", "--label_rate=0.01"]
        # else:
        #     command = ["python", "train_RSGNN.py", f"--edge_rate={params['edge_rate']}", f"--s_rate={params['s_rate']}",
        #                f"--seed={params['seed']}", f"--dataset={params['dataset']}", "--label_rate=0.01"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate()  # 获取子进程的输出和错误
        # 将输出写入文件
#         file.write(f"{params}\n")
#         file.write(stdout)
#         file.write(stderr)
#         file.write('\n\n')
        file.write(f"{params}\n")
        print(stdout, end='')  # 显示在控制台
        file.write(stdout)  # 同时写入文件
        print(stderr, end='')  # 显示在控制台
        file.write(stderr)  # 同时写入文件
        file.write('\n\n')