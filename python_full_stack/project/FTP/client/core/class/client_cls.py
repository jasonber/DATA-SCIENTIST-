import socket
import struct
import os


class Client(socket.socket):
    #? 这是一个客户端需要与服务端通信，所以要继承socket类
    def __init__(self, ip_port):
        #? 需要初始化一个request
        self.request = self.connect(ip_port)
    
    #? 注册
    def regist(self):
        #? 接收输入用户名提示
        print(self.request.recv(1024).decode('utf-8'))
        new_id = input(">>>")
        self.request.send(new_id.encode('utf-8'))
        #? 注册用户名是否重复
        id_res = self.request.recv(1024).decode('utf-8')
        print(id_res)
        while id_res == "用户名重复":
            new_id = input("请输入用户名")
            self.request.send(new_id.encode('utf-8'))
        #? id_res是创建用户名的结果，如果不重复，那么接收输入密码提示
        # print(self.request.recv(1024).decode('utf-8'))
        new_pwd = input(">>>")
        self.request.send(new_pwd.encode('utf-8'))
        #? 接收注册结果
        print(self.request.recv(1024).decode('utf-8'))
    
    #? 登录
    def login(self):
        #? 接收用户名输入提示
        print(self.request.recv(1024).decode('utf-8'))
        user_id = input(">>>")
        self.request.send(user_id.encode('utf-8'))
        #? 接收密码输入提示
        print(self.request.recv(1024).decode('utf-8'))
        user_pwd = input(">>>")
        self.request.send(user_pwd.encode('utf-8'))
        #? 接收登陆提示
        print(self.request.recv(1024).decode('utf-8'))
    

    #? 主页
    def homepage(self):
        print(self.request.recv(1024).decode('utf-8'))
        a = 1
        while a:
            flag = input("1 登录\n2 注册\n>>>")
            self.request.send(flag.encode('utf-8'))
            if flag == "1":
                login.login(self.request)
                a = 0
            elif flag == "2":
                regist.regist(self.request)
                a = 0
            else:
                print("输入错误， 请重新输入")

    #? 文件传输
    @staticmethod
    def pack_header(local_path, module='i'):
        data_size = os.path.getsize(local_path)
        header = struct.pack(module, data_size)
        return header
    
    def unpack_header(self, longth=4, module='i'):
        header = self.request.recv(longth)
        data_size = struct.unpack(module, header)
        return data_size

    def recv_file(self):
        print(self.request.recv(1024).decode('utf-8'))
        file_path = input(">>>")
        self.request.send(file_path.encode('utf-8'))
        save_path = input('请输入本地保存地址:')
        data_size = unpack_header(self.request)[0]
        recv_size = 0
        with open(save_path, 'wb') as f:
            while recv_size < data_size:
                data = self.request.recv(1024)
                f.write(data)
                recv_size += len(data)
        print("接收完成")

    def send_file(self):
        local_path = input('请输入要发送的文件地址:')
        print(self.request.recv(1024).decode('utf-8'))
        save_path = input('>>>')
        self.request.send(save_path.encode('unicode'))
        header = pack_header(local_path)
        self.request.send(header)
        with open(local_path, 'rb') as f:
            for lines in f:
                # data = lines
                self.request.send(lines)
        print("发送完成")

    def trans_data(command, self):
        if command == '发送':
            send_file(self.request)
        if command == '接收':
            recv_file(self.request)

    #? 目录操作指令
    def command_res(command, self):
        #? 不同的命令输出的结果不同，发送与接收的模式也不同
        # ?有结果：成功有结果（1）、，错误报错（1）。无结果：成功无结果（0）
        # ?其中成功无结果无法发送，所以发送指定内容
        # ?发送和接收有们固定的方式
        flag = self.request.recv(3).decode('utf-8')
        # print(a)
        # flag = a.decode('utf-8')

        if command in ['查看', '新建', '删除']:
            #! 这里的传输存在粘包现象，应该是数据过小跟后面的包黏在一起了。
            # !这是不是粘包,是因为上面打印了。但是这里传送数字不太能处理好。
            if flag == "有":
                command_header = self.request.recv(4)
                command_size = struct.unpack('i', command_header)[0]
                recv_size = 0
                res = ""
                while recv_size < command_size:
                    data = self.request.recv(1024).decode('utf-8')
                    res += data
                    recv_size += len(data)
                print(res)
            elif flag == "无":
                #? mkdir 和 rm 都没有结果返回，所以
                print(self.request.recv(12).decode('utf-8'))
        if command == "切换":
            #? 因为切换是要切换的是整个文件的工作目录，所以要使用os.chdir()，
            # ?没有有，且返回结果需要提示工作目录，所以单独接收
            print(self.request.recv(1024).decode('utf-8'))
