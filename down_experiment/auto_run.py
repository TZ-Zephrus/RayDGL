import os
import subprocess
import numpy as np
import time

# os.system('python3 /home/asd/文档/wtz/wtz/RayDGL/down_experiment/reddit.py --part_num=100')

time1 = time.time()
# max_loss = [0,5,10,15,20,25,30,35,40,45,50,55,60,65]
# max_loss = [30,40,60]
max_loss = [0]
train_times = 10
acc_total = np.reshape(np.arange(1,train_times+1), (1, train_times))    # acc矩阵

for i in max_loss:
    acc_line = np.array([])     # acc矩阵的一行（一种part_num)
    for j in range(train_times):
        res = subprocess.Popen('python3 /home/bit/文档/wtz/RayDGL/down_experiment/cora.py --part_num={} --loss_time=0 --dataset=citeseer'.format(i),
                            shell=True,              # 新开一个终端
                            stdout=subprocess.PIPE,  # 执行完命令, 将正确输出放到一个管道里
                            stderr=subprocess.PIPE   # 将错误输出放到一个管道里
                            )
        result = res.stdout.read()          # 拿到的是 bytes 格式的字符
        result= str(result,encoding="gbk")  # 在windows需要使用gbk编码 
        print(result)
        # 找出acc``
        lines = result.split('\n')
        for line in lines:
            if 'test accuracy' in line:
                acc = np.array(float(line[-6:]))
            # else:
            #     print('no_acc')
        acc_line = np.append(acc_line, acc)
    acc_line = np.reshape(acc_line, (1, train_times))
    acc_total = np.append(acc_total, acc_line, axis=0)
print(acc_total) 

acc_final = np.mean(acc_total, axis=1)
print(acc_final)
print('time: ', time.time()-time1)

