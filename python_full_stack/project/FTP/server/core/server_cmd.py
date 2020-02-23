import subprocess
import os
file_path ="/".join(__file__.split('/')[:-2])
#? 保证用户的命令只能在服务器的文件存储位置处操作
usr_path = file_path + '/user_dir'
os.chdir(usr_path)
import trans_data
import struct
import time

def bash(cmd, request):
    proc = subprocess.Popen(cmd, 
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            shell=True)

    res = proc.stdout.read()
    err = proc.stderr.read()
    
    #? 不同的命令输出的结果不同，发送与接收的模式也不同
    # ?有结果：成功有结果（1）、，错误报错（1）。无结果：成功无结果（0）
    # ?其中成功无结果无法发送，所以发送指定内容
    if err:
        #! 这里为什么不能发送数字的编码和解码？
        #! 信息过短 出现了粘包现象，暂时不知道怎么解决
        request.send('有'.encode('utf-8'))
        err_size = len(res)
        header_res = struct.pack('i', err_size)
        request.send(header_res)
        request.send(err)
    elif res: 
        request.send('有'.encode('utf-8'))
        res_size = len(res)
        header_res = struct.pack('i', res_size)
        request.send(header_res)
        request.send(res)
    else:
        request.send('无'.encode('utf-8'))
        request.send("操作成功".encode('utf-8'))


def server_cmd(request):
    command = request.recv(1024).decode('utf-8')
    print(command)
    # time.sleep(2)
    try:
        cmd, directory = command.split(" ")
    except ValueError as v:
        print("接收命令处的错误",v)
        print('''命令错误\n请按照"命令 目录"的格式输入命令''')
    try:
        if cmd == "查看":
            # request.send(os.getcwd().decode('utf-8'))
            cmd_lst = "ls" + " " + directory
            bash(cmd_lst, request)
        elif cmd == "新建":
            cmd_lst = 'mkdir -p' + " " + directory
            bash(cmd_lst, request)
        elif cmd == "删除":
            cmd_lst = "rm" + " " + directory
            bash(cmd_lst, request)
        elif cmd == "切换":
            try:
                # cmd_lst = "cd" + " " + directory
                os.chdir(directory)
                msg = "当前目录：{}".format(os.getcwd())
                request.send(msg.encode('utf-8'))
            except Exception as e:
                print(e)
                request.send(e.encode('utf-8'))
        elif cmd == '发送':
            trans_data.recv_file(request)
        elif cmd == "接收":
            trans_data.send_file(request)
    except UnboundLocalError as u: 
        # print("执行命令时的错误", u)
        pass

def run():
    command = input('请输入命令:') #? 有四个命令：查看、新建、删除、切换
    server_cmd(command)

if __name__ == "__main__":
    while 1:
       run()
