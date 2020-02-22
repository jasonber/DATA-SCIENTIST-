def homepage(request):
    print(request.recv(1024).decode('utf-8'))
    flag = input("1 登录\n2 注册\n>>>")
    request.send(flag.encode('utf-8'))