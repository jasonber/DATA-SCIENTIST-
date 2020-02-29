import hashlib
import os
file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
import login
import regist
import server_cmd


# !将网络服务与功能分开就可以了
# !尽力了


class Action:

    def __init__(self, request):
        self.request = request

    #? 服务器的运行流程, 继承于BaseRequestHandler
    #? 这里定义了服务实际运行的流程
    def handle(self):
        #? 进入主页
        self.homepage(self.request)
        #? 与客户端交互
        while 1:
            self.run_command(self.request)

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

    
    # todo 命令执行
    def run_cmd(self):
        server_cmd.run_cmd(self.request)


    
