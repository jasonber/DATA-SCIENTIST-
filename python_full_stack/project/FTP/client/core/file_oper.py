import socket
import json
import os
import struct

print("文件地址",__file__)
file_path = "/".join(__file__.split('/')[:-1])
print("修改文件地址", file_path)
os.chdir(file_path)


def pack_header(path, module='i'):
    data_size = os.path.getsize(path)
    header = struct.pack(module, data_size)
    return header

def unpack_header(request, longth=4, module='i'):
    header = request.recv(longth)
    data_size = struct.unpack(module, header)
    return data_size


def recv_file(request):
    print(request.recv(1024).decode('utf-8'))
    file_path = input(">>>")
    request.send(file_path.encode('utf-8'))
    save_path = input('请输入本地保存地址:')
    data_size = unpack_header(request)[0]
    recv_size = 0
    with open(save_path, 'wb') as f:
        while recv_size < data_size:
            data = request.recv(1024)
            f.write(data)
            recv_size += len(data) 
    print("接收完成, OK")

def send_file(request):
    local_path = input('请输入要发送的文件地址:')
    print(request.recv(1024).decode('utf-8'))
    save_path = input('>>>')
    request.send(save_path.encode('unicode'))
    header = pack_header(local_path)
    request.send(header)
    with open(local_path, 'rb') as f:
        for lines in f:
            # data = lines
            request.send(lines)        
    print("发送完成")

def get_settings():
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line
    setting_dic = json.loads(setting_json)
    return setting_dic    

def trans_data(command, request):
    if command == '发送':
        send_file(request)
    if command == '接收':
        recv_file(request)