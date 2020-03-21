import socket
import json
import os
import sys

file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
os.getcwd()

def get_settings():
    #? 读取配置文件
    setting_json = ""
    with open("../conf/settings.json", "r", encoding="utf-8") as f:
        for line in f.read():
            setting_json += line
    setting_dic = json.loads(setting_json)
    return setting_dic

#? 配置连接信息
setting_dic = get_settings()
IP_PORT = (setting_dic["IP"], setting_dic['PORT'])
client = socket.socket()
client.connect(IP_PORT)

#? 开始聊天

#? 循环聊天
while True:
    flag = 0
    if flag == 0:
        data = client.recv(1024).decode('utf-8')
        print(data)
        flag += 1
    data = input('>>>>')
    client.send(data.encode('utf-8'))
    if data == "exit":
        break
    else:
        data = client.recv(1024).decode("utf-8")
        print(data)
client.close()
print('over')

#
# if __name__ == '__main__':
#     working()