import socket
import json
import os
import struct
file_path = '/'.join(__file__.split(r'/')[:-1])
os.chdir(file_path)
import sys
sys.path.append(file_path)
from client_act import Action

HOME_OPT = {'登录': 'login', '注册': 'regist'}

CMD_CODE = {
    '查看': 'check_dir',
    '新建': 'make_dir',
    '删除': 'dele_dir',
    '切换': 'swithc_dir',
    '上传': 'uplod',
    '下载': 'download'
}


def run():
    #? 获取配置
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line

    setting_dic = json.loads(setting_json)
    IP_PORT = (setting_dic['IP'], setting_dic['PORT'])

    #? 连接服务器
    client = socket.socket()
    client.connect(IP_PORT)

    # todo 主页
    act = Action(client.request)
    print(act.recv(1024).decode('utf-8'))
    a = 1
    while a:
        home_flag = ['登录', '注册']
        for num, i in enumerate(home_flag, 1):
            print("%s %s" % (num, i))
        flag = input("请选择1 or 2：")
        act.send(flag.encode('utf-8'))
        if hasattr(Action, HOME_OPT[home_flag[flag-1]]):
            getattr(Action, HOME_OPT[home_flag[flag-1]])()
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

        client.send(command.encode('utf-8'))
        act.run_cmd(command)

        if cmd == '离开':
            break

    client.close()


if __name__ == '__main__':
    run()