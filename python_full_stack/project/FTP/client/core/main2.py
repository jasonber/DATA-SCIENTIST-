import socket
import json
import os
import struct
file_path = '/'.join(__file__.split(r'/')[:-1])
os.chdir(file_path)
import sys
sys.path.append(file_path)
import file_oper
import homepage
import cmd_res


setting_json = ""
with open('../conf/settings.json', 'r', encoding='utf-8') as f:
    for line in f.read():
        setting_json += line

setting_dic = json.loads(setting_json)
IP_PORT = (setting_dic['IP'], setting_dic['PORT'])

client = socket.socket()
client.connect(IP_PORT)

homepage.homepage(client)
while 1:
    command = input('请输入命令：')
    client.send(command.encode('utf-8'))
    if command in ['接收', '发送']:
        file_oper.trans_data(command, client)
    else:
        cmd_res.cmd_res(client)






    # #! 大文件的传输需要注意只能用bytes传递，所以不存在decode。
    # if command == '接收':
    #     header = client.recv(4)
    #     data_size = struct.unpack('i', header)[0]
    #     recv_size = 0
    #     with open(local_path, 'wb') as f:
    #         while recv_size < data_size:
    #             data = client.recv(1024)
    #             f.write(data)
    #             recv_size += len(data)
    #     print(b"download,OK")

    # if command == '发送':
    #     with open(r'D:\个人目标\my_git_hub\DATA-SCIENTIST-\python_full_stack\project\FTP\client\DB\upload\test_up', 'rb') as f:
    #         data = f.read()
    #     client.send(data)        
    #     print(b"upload OK")