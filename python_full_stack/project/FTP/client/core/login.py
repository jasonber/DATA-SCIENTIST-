def login(request):
    #? 接收用户名输入提示
    print(request.recv(1024).decode('utf-8'))
    user_id = input(">>>")
    #? 接收密码输入提示
    print(request.recv(1024).decode('utf-8'))
    user_pwd = input(">>>")
    #? 接收登陆提示
    print(request.recv(1024).decode('utf-8'))