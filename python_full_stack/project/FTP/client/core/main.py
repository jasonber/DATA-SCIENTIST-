import socket
import json
import os
import struct
file_path = '/'.join(__file__.split(r'/')[:-1])
os.chdir(file_path)
import sys
sys.path.append(file_path)
import client_cls

def run():  
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line

    setting_dic = json.loads(setting_json)
    IP_PORT = (setting_dic['IP'], setting_dic['PORT'])

    client = client_cls.Client()
    client.connect(IP_PORT)

    client.homepage()
    while 1:
        command = input('请输入命令：')
        # ?验证命令的格式是否正确
        try:
            cmd, directory = command.split(" ")
        except ValueError:
            print('''命令错误\n请按照"命令 目录"的格式输入命令''')
            continue
        if cmd not in ['查看', '新建', '删除', '切换', '接收', '传送']:
            print('命令错误')
            continue

        client.send(command.encode('utf-8'))

        client.run_cmd(command)

if __name__ == '__main__':
    run()