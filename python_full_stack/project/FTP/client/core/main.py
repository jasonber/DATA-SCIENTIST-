import socket
import json
import os
import struct
file_path = '/'.join(__file__.split(r'/')[:-1])
os.chdir(file_path)
import sys
sys.path.append(file_path)
from client_act import Action


def run():
    # todo 获取配置
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line
    setting_dic = json.loads(setting_json)
    IP_PORT = (setting_dic['IP'], setting_dic['PORT'])
    CMD_CODE = setting_dic["CMD_CODE"]

    # todo 连接服务器
    act = Action()
    act.connect(IP_PORT)

    # todo 主页
    # act = Action(client)
    print(act.recv(1024).decode('utf-8'))
    a = 1
    while a:
        home_flag = ['登录', '注册']
        for num, i in enumerate(home_flag, 1):
            print("%s %s" % (num, i))
        flag = input("请选择1 or 2：")
        act.sendall(flag.encode('utf-8'))
        if hasattr(Action, CMD_CODE[home_flag[int(flag) - 1]]):
            login_res = getattr(Action, CMD_CODE[home_flag[int(flag) - 1]])(act) #! 用对象当作参数，等于将self传入
            if login_res == "0030":
                print("登录成功")
            else:
                print("用户名或密码错误，请重新输入")
                continue
            a = 0
        else:
            print("输入错误， 请重新输入")

        

    while 1:
        command = input('请输入命令：')
        # todo 验证命令的格式是否正确
        try:
            cmd, directory = command.split(" ")
        except ValueError:
            print('''命令错误\n请按照"命令 目录"的格式输入命令''')
            continue

        if cmd not in ['查看', '新建', '删除', '切换', '上传', '下载', '离开']:
            print('命令错误')
            continue

        act.send(command.encode('utf-8'))
        act.run_cmd(cmd)

        if cmd == '离开':
            break

    act.close()


if __name__ == '__main__':
    run()