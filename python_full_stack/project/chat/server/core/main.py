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

def set_server(setting_dic, lisetn=5):
    #? 配置服务器信息
    IP_PORT = (setting_dic["IP"], setting_dic['PORT'])
    server = socket.socket()
    server.bind(IP_PORT)
    server.listen(5)
    print(setting_dic["WORKING_MSG"])
    return server

def chat(request, client_address):
    #? 开始聊天
    flag = 0
    if flag == 0:
        request.send('连接成功'.encode('utf-8'))
        flag += 1 
    data = request.recv(1024).decode("utf-8")
    request.send('收到'.encode("utf-8"))
    return data, request

def working():
    #? 循环聊天
    setting_dic = get_settings()
    server = set_server(setting_dic)
    while True:
        request, client_address = server.accept()
        while True:
            data, request = chat(request, client_address)
            if data == "exit":
                break
            else:
                print(data)
        request.close()
        print('over')


if __name__ == '__main__':
    working()






#? 结束
# request.close()
# server.close()
