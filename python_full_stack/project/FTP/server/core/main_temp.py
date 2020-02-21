import socket
import json
import os
import struct

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
        # try:
        if command == 'put':
            with open('/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/test_down1111', 'w') as f:
                data = request.recv(1024).decode('utf-8')
                f.write(data)
            print(b"upload,OK")

        if command == 'pull':
            #? 解决粘包
            data_size = os.path.getsize('/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/backiee-118342.jpg')
            header = struct.pack('i', data_size)
            request.send(header)
            with open('/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/backiee-118342.jpg', 'rb') as f:
                for lines in f:
                    # data = lines
                    request.send(lines)        
            print(b"download OK")
        # except Exception as e:
        #     print(e)
        #     continue


def pack_header(path, module='i', oper=1):
    data_size = os.path.getsize(path)
    header = struct.pack(module, data_size)
    return header

def unpack_header(request, longth=4):
    header = request.recv(longth)
    data_size = struct.unpack(header)
    return data_size

def trans_data(command, path, request):
    if command == '接收':
        send_file(path, request)

    if command == '发送':
        recv_file(path, request)

def recv_file(path, request):
    data_size = upack_header(request)
    recv_size = 0
    with open(path, 'wb') as f:
        while recv_size < data_size:
            data = request.recv(1024)
            f.write(data)
            recv_siez += len(data) 
    print("接收完成, OK")

def send_file(path, request):
    header = get_header(path)
    request.send(header)
    with open(path, 'rb') as f:
        for lines in f:
            # data = lines
            request.send(lines)        
    print(b"发送完成")

def get_settings():
    file_path = '/'.join(__file__.spliit('/')[:-1])
    os.chdir(file_path)

    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line
    setting_dic = json.loads(setting_json)
    return setting_dic    

