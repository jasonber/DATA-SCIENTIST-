import struct
import os
import file_oprt
import client_cmd


class Action:
    CMD_CODE = {
        '查看': 'check_dir',
        '新建': 'make_dir',
        '删除': 'dele_dir',
        '切换': 'swithc_dir',
        '上传': 'upload',
        '下载': 'download'
    }

    def __init__(self, request):
        # todo 获得通信的能力
        self.request = request


    # todo 注册
    def regist(self):
        #! 因为注册要验证用户名是否重复，所以不能打包注册信息
        # todo 接收输入用户名提示
        new_id = input("请输入用户名：")
        self.request.send(new_id.encode('utf-8'))
        # todo 注册用户名是否重复
        id_res = self.request.recv(1024).decode('utf-8')
        print(id_res)
        while id_res == "用户名重复":
            new_id = input("请输入用户名")
            self.request.send(new_id.encode('utf-8'))
        new_pwd = input("请输入密码")
        self.request.send(new_pwd.encode('utf-8'))
        # todo 接收注册结果
        print(self.request.recv(1024).decode('utf-8'))

    # todo 登录
    def login(self):
        # todo 输入用户名和密码
        user_id = input("请输入用户名：")
        user_pwd = input("请输入密码：")
        # todo 打包用户信息
        user_info = user_id + " " + user_pwd
        self.request.send(user_info.encode('utf-8'))
        # todo 接收登录结果
        print(self.request.recv(1024).decode('utf-8'))

    # todo 命令执行
    def run_cmd(self, command):
        if command in ['上传', '下载']:
            hasattr(client_cmd, CMD_CODE[command])(self.request)
        else:
            client_cmd.cmd_res(command, self.request)
