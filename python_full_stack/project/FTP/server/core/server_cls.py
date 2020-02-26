import socketserver
import hashlib
import os
file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
import login
import regist
import server_cmd


class Server(socketserver.BaseRequestHandler):
    
    def __init__(self):
        self.request, self.address = super().get_request()

    #? 服务器的运行流程, 继承于BaseRequestHandler
    #? 这里定义了服务实际运行的流程
    def handle(self):
        #? 进入主页
        self.homepage(self)
        #? 与客户端交互
        while 1:
            self.run_command(self)

    #? 登录
    @staticmethod
    @property
    def __get_user_data():
        user_data = {}
        with open(r'../DB/user', 'r', encoding='utf-8', newline=None) as f:
            for line in f.readlines():
                user_id, user_pwd = line.split(" ")
                user_data[user_id] = user_pwd[:-1]
        return user_data

    def login(self):
        user_data = self.__get_user_data()
        user_id = login.verify(user_data, self.request)
        login.__switch_usr_dir(user_id)

    #? 注册
    def regist(self):
        regist.regist(self.request)

    #? 操作目录与文件传输
    def run_command(self):
        server_cmd.run_cmd(self.request)

    #? 主页
    def homepage(self):
        self.request.send("欢迎光临".encode('utf-8'))
        flag = self.request.recv(4).decode('utf-8')
        if flag == '1':
            login.login(self.request)
        elif flag == '2':
            regist.regist(self.request)
    
if __name__ == '__main__':
    test_ser = socketserver.ThreadingTCPServer(("192.168.3.9", 8000), Server)
