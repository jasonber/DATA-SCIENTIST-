import struct
import socket
import json
import os
import struct
file_path = "/".join(__file__.split('/')[:-1])
os.chdir(file_path)


# todo 接收命令执行的结果
def cmd_res(cmd, request):
    # ? 不同的命令输出的结果不同，发送与接收的模式也不同
    # ?有结果：成功有结果（0001）、，错误报错（0001）。无结果：成功无结果（0000）
    # ?其中成功无结果无法发送，所以发送指定内容
    # ?发送和接收有他们固定的方式
    flag = request.recv(4).decode('utf-8')
    # print(a)
    # flag = a.decode('utf-8')

    if cmd in ['查看', '新建', '删除']:
        if flag == "0001":
            cmd_header = request.recv(4)
            cmd_size = struct.unpack('i', cmd_header)[0]
            recv_size = 0
            res = ""
            while recv_size < cmd_size:
                data = request.recv(1024).decode('utf-8')
                res += data
                recv_size += len(data)
            print(res)
        elif flag == "0000":
            #? mkdir 和 rm 都没有结果返回，所以
            print(request.recv(12).decode('utf-8'))
    if cmd == "切换":
        #? 因为切换是要切换的是整个文件的工作目录，所以要使用os.chdir()，
        # ?没有有，且返回结果需要提示工作目录，所以单独接收
        print(request.recv(1024).decode('utf-8'))


# todo 数据传输
def pack_header(path, module='i'):
    data_size = os.path.getsize(path)
    header = struct.pack(module, data_size)
    return header


def unpack_header(request, longth=4, module='i'):
    header = request.recv(longth)
    data_size = struct.unpack(module, header)
    return data_size

#! 断点续传
def renewal():
    pass

def recv_data(request, data_size, save_path, mod='ab'):
    recv_size = 0
    with open(save_path, 'wb') as f:
        while recv_size < data_size:
            data = request.recv(1024)
            f.write(data)
            recv_size += len(data)
    print("接收完成")


def send_data(request, send_path, mod='rb'):
    with open(send_path, 'rb') as f:
        for lines in f:
            # data = lines
            request.send(lines)
    print("发送完成")


def download(request):
    server_path = input('请输入要下载的文件地址:')
    local_path = input('请输入文件的保存地址:')
    request.send(server_path.encode('utf-8'))
    data_size = unpack_header(request)[0]
    recv_data(request, data_size, local_path)


def upload(request):
    local_path = input('请输入要发送的文件地址:')
    server_path = input('请输入文件的保存地址:')
    request.send(server_path.encode('unicode'))
    header = pack_header(local_path)
    request.send(header)
    send_data(request, local_path)


def get_settings():
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line
    setting_dic = json.loads(setting_json)
    return setting_dic
