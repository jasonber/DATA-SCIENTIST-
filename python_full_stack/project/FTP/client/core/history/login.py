def login(request):
    #? 接收用户名输入提示
    print(request.recv(1024).decode('utf-8'))
    user_id = input(">>>")
    request.send(user_id.encode('utf-8'))
    #? 接收密码输入提示
    print(request.recv(1024).decode('utf-8'))
    user_pwd = input(">>>")
    request.send(user_pwd.encode('utf-8'))
    #? 接收登陆提示
    print(request.recv(1024).decode('utf-8'))