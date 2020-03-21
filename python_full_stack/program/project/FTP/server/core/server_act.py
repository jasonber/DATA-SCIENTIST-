import hashlib
import socketserver
import os
file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
import login
import regist
import server_cmd

# !将网络服务与功能分开就可以了
# !尽力了
HOME_OPT = {'登录': 'login', '注册': 'regist'}

user_data = {}
with open(r'../DB/user', 'r', encoding='utf-8', newline=None) as f:
    for line in f.readlines():
        user_id, user_pwd = line.split(" ")
        user_data[user_id] = user_pwd[:-1]


class Action():


    def __init__(self, request, addr):
        self.request = request
        self.addr = addr
        # super(Action, self).__init__()

    # def handle(self):
    #         # ! 服务器的工作内容
    #     act = Action(self.request)
    #     # todo 主页
    #     self.request.send("欢迎光临".encode('utf-8'))
    #     flag = int(self.request.recv(4).decode('utf-8'))
    #     home_flag = ['登录', '注册']
    #     if hasattr(Action, HOME_OPT[home_flag[flag - 1]]):
    #         getattr(Action, HOME_OPT[home_flag[flag - 1]])(self)
    #     #? 与客户端交互
    #     while 1:
    #         act.run_cmd()

    #? 服务器的运行流程, 继承于BaseRequestHandler
    #? 这里定义了服务实际运行的流程

    #? 登录

    def login(self):
        user_id = login.verify(user_data, self.request)
        login.switch_usr_dir(user_id)

    #? 注册
    def regist(self):
        regist.regist(self.request)

    # todo 命令执行
    def run_cmd(self):
        server_cmd.run_cmd(self.request)
