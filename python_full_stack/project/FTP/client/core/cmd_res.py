import struct

def cmd_res(cmd, request):
    #? 不同的命令输出的结果不同，发送与接收的模式也不同
    # ?有结果：成功有结果（1）、，错误报错（1）。无结果：成功无结果（0）
    # ?其中成功无结果无法发送，所以发送指定内容
    # ?发送和接收有他们固定的方式
    flag = request.recv(3).decode('utf-8')
    # print(a)
    # flag = a.decode('utf-8')
    
    if cmd in ['查看', '新建', '删除']:
        #! 这里的传输存在粘包现象，应该是数据过小跟后面的包黏在一起了
        if flag == "有":   
            cmd_header = request.recv(4)
            cmd_size = struct.unpack('i', cmd_header)[0]
            recv_size = 0
            res = ""
            while recv_size < cmd_size:
                data = request.recv(1024).decode('utf-8')
                res += data
                recv_size += len(data)
            print(res)
        elif flag == "无":
            #? mkdir 和 rm 都没有结果返回，所以
            print(request.recv(12).decode('utf-8'))
    if cmd == "切换":
        #? 因为切换是要切换的是整个文件的工作目录，所以要使用os.chdir()，
        # ?没有有，且返回结果需要提示工作目录，所以单独接收
        print(request.recv(1024).decode('utf-8'))
