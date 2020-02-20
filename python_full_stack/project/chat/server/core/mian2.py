import socket
import json
import os
import sys

file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
os.getcwd()


def get_settings():
    # ? 读取配置文件
    setting_json = ""
    with open("../conf/settings.json", "r", encoding="utf-8") as f:
        for line in f.read():
            setting_json += line
    setting_dic = json.loads(setting_json)
    return setting_dic

setting_dic = get_settings()
# ? 配置服务器信息
IP_PORT = (setting_dic["IP"], setting_dic['PORT'])
server = socket.socket()
server.bind(IP_PORT)
server.listen(5)
print(setting_dic["WORKING_MSG"])


# ? 开始聊天



# ? 循环聊天
while True:
    #! 这里会阻塞，这里的含义是服务始终等待不同的用户接入
    request, client_address = server.accept()
    while True:
        #! 这里才是真正的聊天
        flag = 0
        if flag == 0:
            request.send('连接成功'.encode('utf-8'))
            flag += 1
        data = request.recv(1024).decode("utf-8")
        request.send('收到'.encode("utf-8"))
        if data == "exit":
            break
        else:
            print(data)

request.close()
print('over')


# if __name__ == '__main__':
#     working()
