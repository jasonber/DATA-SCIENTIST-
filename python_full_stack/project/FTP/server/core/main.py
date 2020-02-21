import socket
import json
import os

# file_path = '/'.join(__file__.spliit('/')[:-1])
# os.chdir(file_path)

setting_json = ""
with open('/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/conf/settings.json', 'r', encoding='utf-8') as f:
    for line in f.read():
        setting_json += line

setting_dic = json.loads(setting_json)
IP_PORT = (setting_dic['IP'], setting_dic['PORT'])

server = socket.socket()
server.bind(IP_PORT)
server.listen(5)
print('Server is working.......')


while True:
    #? 上传与下载
    request, client_address = server.accept()
    print('连接成功')
    while True:
        command = request.recv(1024).decode('utf-8')
        #?
        try:
            if command == 'put':
                with open('/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/test_down1111', 'w') as f:
                    data = request.recv(1024).decode('utf-8')
                    f.write(data)
                print(b"upload,OK")

            if command == 'pull':
                with open('/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/test_up', 'rb') as f:
                    data = f.read()
                request.send(data)        
                print(b"download OK")
        except Exception as e:
            print(e)
            continue