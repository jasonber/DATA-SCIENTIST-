import socket
import json
import os

# file_path = '/'.join(__file__.spliit('/')[:-1])
# os.chdir(file_path)

setting_json = ""
with open(r'D:\个人目标\my_git_hub\DATA-SCIENTIST-\python_full_stack\project\FTP\server\conf\settings.json', 'r', encoding='utf-8') as f:
    for line in f.read():
        setting_json += line

setting_dic = json.loads(setting_json)
IP_PORT = (setting_dic['IP'], setting_dic['PORT'])

client = socket.socket()
client.connect(IP_PORT)


#? 上传与下载
while True:
    command = input('>>>>')
    client.send(command.encode('utf-8'))

    #?
    if command == 'pull':
        with open(r'D:\个人目标\my_git_hub\DATA-SCIENTIST-\python_full_stack\project\FTP\client\DB\download\test_down11111', 'w') as f:
            data = client.recv(1024).decode('utf-8')
            f.write(data)
        print(b"download,OK")

    if command == 'put':
        with open(r'D:\个人目标\my_git_hub\DATA-SCIENTIST-\python_full_stack\project\FTP\client\DB\upload\test_up', 'rb') as f:
            data = f.read()
        client.send(data)        
        print(b"upload OK")