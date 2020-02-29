import subprocess
import os
file_path = "/".join(__file__.split('/')[:-2]) # * 保证用户的命令只能在服务器的文件存储位置处操作
usr_path = file_path + '/user_dir'
os.chdir(usr_path)
import struct
import time
import sys
import json

CMD_CODE = {
    '查看': 'check_dir',
    '新建': 'make_dir',
    '删除': 'dele_dir',
    '切换': 'swithc_dir',
    '上传': 'upload',
    '下载': 'download'
}


def send_msg(request, msg):
    request.send(b'0001')
    err_size = len(msg)
    header_res = struct.pack('i', err_size)
    request.send(header_res)
    request.send(msg)


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
        send_msg(request, err)
    elif res:
        send_msg(request, res)
    else:
        #? 某些命令执行成功没有结果
        request.send(b'0000')
        request.send("操作成功".encode('utf-8'))


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
        request.send(msg.encode('utf-8'))
    except Exception as e:
        print(e)
        request.send(e.encode('utf-8'))


# todo 数据传输
def pack_header(path, module='i', oper=1):
    data_size = os.path.getsize(path)
    header = struct.pack(module, data_size)
    return header


def unpack_header(request, longth=4, module='i'):
    header = request.recv(longth)
    data_size = struct.unpack(module, header)
    return data_size


##! 断点续传
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


def upload(request, directory):
    local_path = request.recv(1024).decode('unicode')
    data_size = unpack_header(request)[0]
    recv_data(request, data_size, local_path)


def download(request, directory):
    local_path = request.recv(1024).decode('utf-8')
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


def run_cmd(request):
    command = request.recv(1024).decode('utf-8')
    print(command)
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
        # print("执行命令时的错误", u)
        pass



