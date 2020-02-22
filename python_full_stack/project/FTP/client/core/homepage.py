import os 
file_path = '/'.join(__file__.split('/'))
os.chdir(file_path)

import login
import regist



def homepage(request):
    print(request.recv(1024).decode('utf-8'))
    a = 1
    while a:
        flag = input("1 登录\n2 注册\n>>>")
        request.send(flag.encode('utf-8'))
        if flag == "1":
            login.login(request)
            a = 0
        elif flag == "2":
            regist.regist(request)
            a = 0
        else:
            print("输入错误， 请重新输入")