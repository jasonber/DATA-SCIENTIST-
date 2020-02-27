import os
file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path) 

import login 
import regist

def homepage(request):
    request.send("欢迎光临".encode('utf-8'))
    flag = request.recv(4).decode('utf-8')
    if flag == '1':
        login.login(request)
    elif flag == '2':
        regist.regist(request)