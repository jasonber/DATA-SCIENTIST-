import struct
import socket
import json
import struct
import shutil
import sys
import hashlib
import os
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


#! 进度条
def progress_bar(total_size, recv_size):
    per = recv_size / total_size * 100
    print("\r进度:{sign} {per:.2f}".format(sign="*" * int(per), per=per), end="")


def md5_func(string):
    # todo md5 加密
    md5_func = hashlib.md5()
    md5_func.update(string)
    return md5_func.hexdigest()


def consistency(path):
    md5_func = hashlib.md5()
    with open(path, 'rb') as f:
        for line in f.read():
            md5_func.update(line)
    return md5_func.hexdigest()


def recv_data(request, save_path, total_size, recv_size=0, mod='ab'):
    try:
        md5_obj = hashlib.md5()
        f = open(save_path, mode='wb')
        while recv_size < total_size:
            data = request.recv(1024)
            f.write(data)
            md5_obj.update(data)
            f.flush()
            progress_bar(recv_size, total_size)
            recv_size += len(data)
        md5_val = md5_obj.hexdigest()
    finally:
        f.close()
    print("接收完成")
    return consistency(save_path)


def send_data(request, send_path, exists_size=0, mod='rb'):
    try:
        f = open(send_path, mode='rb')
        f.seek(exists_size)
        for line in f.read():
            request.sendall(line)
    finally:
        f.close()
    print("发送完成")
    return consistency(send_path)


def download(request, user_id):
    '''
    传输的文件信息的，文件名 大小 传输状态码 用户名 文件路径 服务器地址 服务地址+用户的md5值 已有大小（续传使用）
    file_info = {file_name, file_size, trans_code, user_name, server_path}，有了这些信息才能判断走哪个分支
    服务器的路径 server_path, 客户端的路径 client_path
    '''

    server_path = input('请输入要下载的文件地址:')
    client_path = input('请输入文件的保存地址:')

    file_name = os.path.basename(server_path)
    # request.sendall(server_path.encode('utf-8'))
    file_md5 = md5_func(user_id + server_path)
    file_info["file_md5"] = file_md5
    #! 传输规范
    file_info = {
        "file_name": file_name,
        "server_path": server_path,
        "user_id": user_id
    }
    # data_size = unpack_header(request)[0]
    save_dir = client_path.split('/')[:-1]
    save_path = '/'.join(save_dir.append(file_info['file_md5']))
    exists = os.path.exists(save_path)
    if not exists:
        #? 0010代表文件不存在， 0011代表文件已存在
        file_info['trans_code'] = "0010"
        request.sendall(json.dumps(file_info).encode('utf-8'))
        file_info = json.loads(request.recv(1024).decode('utf-8'))
        total_size = file_info['total_size']
        consis_md5 = recv_data(request, save_path, total_size)
        shutil.move(save_path, client_path)
    else:
        #! 断点续传
        # todo 发送本地文件的大小
        # todo 服务器核对
        # todo 从断开处发送
        # todo 修改临时文件名称
        file_info['trans_code'] = "0011"
        exists_size = os.path.getsize(save_path)
        file_info['exists_size'] = exists_size
        request.sendall(json.dumps(file_info).encode('utf-8'))
        file_info = json.loads(request.recv(1024).decode('utf-8'))
        total_size = file_info['total_size']
        consis_md5 = recv_data(request,
                               save_path,
                               total_size,
                               recv_size=exists_size)
        shutil.move(save_path, client_path)

    #! 一致性检验
    request.sendall(consis_md5.encode('utf-8'))
    trans_code = request.recv(1024).decode('utf-8')
    if trans_code == '0020':
        print('文件一致性检验成功')
    else:
        print('文件一致性检验失败')


def upload(request, user_id):
    client_path = input('请输入要发送的文件地址:')
    server_path = input('请输入文件的保存地址:')
    # ! 传输规范
    file_name = os.path.basename(client_path)
    # request.sendall(server_path.encode('utf-8'))
    file_md5 = md5_func(user_id + client_path)
    total_size = os.path.getsize(client_path)

    file_info = {
        "file_name": file_name,
        "server_path": server_path,
        "user_id": user_id,
        "total_size": total_size,
        "file_md5": file_md5
    }

    request.sendall(json.dumps(file_info).encode('utf-8'))
    file_info = json.loads(request.recv(1024).decode('utf-8'))
    trans_code = file_info['trans_code']
    if trans_code == "0010":
        consis_md5 = send_data(request, server_path)

    if trans_code == "0011":
        exists_size = file_info['exists_size']
        consis_md5 = send_data(request, server_path, exists_size=exists_size)

    #! md5 一致性检验
    send_consis_md5 = request.recv(1024).decode('utf-8')
    if send_consis_md5 == consis_md5:
        #? 0020 表示一致性校验成功，0021表示失败
        request.sendall(b'0020')
    else:
        request.sendall(b'0021')


def get_settings():
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line
    setting_dic = json.loads(setting_json)
    return setting_dic
