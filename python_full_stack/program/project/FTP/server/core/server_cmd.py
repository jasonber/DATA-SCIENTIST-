import subprocess
import struct
import time
import sys
import json
import hashlib
import shutil
import os
file_path = "/".join(__file__.split('/')[:-2])  # * 保证用户的命令只能在服务器的文件存储位置处操作
usr_path = file_path + '/user_dir'
os.chdir(usr_path)

CMD_CODE = {
    '查看': 'check_dir',
    '新建': 'make_dir',
    '删除': 'dele_dir',
    '切换': 'swithc_dir',
    '上传': 'upload',
    '下载': 'download'
}


def sendall_msg(request, msg):
    request.sendall(b'0001')
    err_size = len(msg)
    header_res = struct.pack('i', err_size)
    request.sendall(header_res)
    request.sendall(msg)


def bash(cmd, request):
    # todo 执行目录操作命令
    # * 不同的命令输出的结果不同，发送与接收的模式也不同
    # * 有结果：成功有结果（1）、，错误报错（1）。无结果：成功无结果（0）
    # * 其中成功无结果无法发送，所以发送指定内容

    proc = subprocess.Popen(cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)
    res = proc.stdout.read()
    err = proc.stderr.read()

    if err:
        sendall_msg(request, err)
    elif res:
        sendall_msg(request, res)
    else:
        #? 某些命令执行成功没有结果
        request.sendall(b'0000')
        request.sendall("操作成功".encode('utf-8'))


# todo 目录操作指令
def check_dir(directory, request):
    cmd_lst = "ls" + " " + directory
    bash(cmd_lst, request)


def make_dir(directory, request):
    cmd_lst = 'mkdir -p' + " " + directory
    bash(cmd_lst, request)


def dele_dir(directory, request):
    cmd_lst = "rm" + " " + directory
    bash(cmd_lst, request)


def chng_dir(directory, request):
    try:
        os.chdir(directory)
        msg = "当前目录：{}".format(os.getcwd())
        request.sendall(msg.encode('utf-8'))
    except Exception as e:
        print(e)
        request.sendall(e.encode('utf-8'))


# todo 数据传输
def pack_header(data_size, module='i', oper=1):
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


def upload(request, directory):
    file_info = json.loads(request.recv(1024).decode('utf-8'))
    server_path = file_info['server_path']

    save_dir = server_path.split('/')[:-1]
    save_path = '/'.join(save_dir.append(file_info['file_md5']))
    exsits = os.paht.exists(save_path)

    if not exsits:
        file_info['trans_code'] = "0010"
        request.sendall(json.dumps(file_info).encode('utf-8'))
        file_info = json.loads(request.recv(1024).decode('utf-8'))
        total_size = file_info['total_size']
        consis_md5 = recv_data(request, save_path, total_size)
        shutil.move(save_path, server_path)
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
        shutil.move(save_path, server_path)

    #! 一致性检验
    request.sendall(consis_md5.encode('utf-8'))
    trans_code = request.recv(1024).decode('utf-8')
    if trans_code == '0020':
        print('文件一致性检验成功')
    else:
        print('文件一致性检验失败')


def download(request, directory):
    #! 传输规范
    file_info = json.loads(request.recv(1024).decode('utf-8'))
    user_id = file_info['user_id']
    server_path = file_info['server_path']
    trans_code = file_info['trans_code']
    total_size = os.path.getsize(server_path)
    file_info["total_size"] = total_size
    request.sendallall(json.dumps(file_info).encode('utf-8'))

    if trans_code == "0010":
        consis_md5 = send_data(request, server_path)

    if trans_code == "0011":
        exists_size = file_info['exists_size']
        consis_md5 = send_data(request, server_path, exists_size=exists_size)

    #! md5 一致性检验
    send_consis_md5 = request.recv(1024).decode('utf-8')
    if send_consis_md5 == consis_md5:
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


def run_cmd(request):
    command = request.recv(1024).decode('utf-8')
    # import time
    # print(command)
    # time.sleep(5)
    my_file = sys.modules[__name__]
    try:
        cmd_key, directory = command.split(" ")
    except ValueError as v:
        print("接收命令处的错误", v)
        print('''命令错误\n请按照"命令 目录"的格式输入命令''')
    try:
        if hasattr(my_file, CMD_CODE[cmd_key]):
            getattr(my_file, CMD_CODE[cmd_key])(request, directory)
    except UnboundLocalError as u:
        print("执行命令时的错误", u)
        pass
    if command == '离开':
        request.close()
