import os
import hashlib

file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)


# user = {}
# with open(r'/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/user', 'r', encoding='utf-8', newline=None) as f:
#     for line in f.readlines():
#         user_id, user_pwd = line.split(" ")
#         user[user_id] = user_pwd[:-1]

# user_id = input('请输入用户名：')
# user_pwd = input('请输入密码：')

# #! 加密密码
# usr_md5 = hashlib.md5()
# usr_md5.update(user_pwd.encode('utf-8'))
# md_pwd = usr_md5.hexdigest()
# if user_id in user.keys() and md_pwd == user[user_id]:
#     print('登录成功')
# else:
#     print('用户名或秘密错误，请重新输入')

def get_user_data():
    user_data = {}
    with open(r'../DB/user', 'r', encoding='utf-8', newline=None) as f:
        for line in f.readlines():
            user_id, user_pwd = line.split(" ")
            user_data[user_id] = user_pwd[:-1]
    return user_data

def verify(user, request):
    request.send("请输入用户名：".encode('utf-8'))
    user_id = request.recv(1024).decode('utf-8')
    request.send("请输入密码：".encode('utf-8'))
    user_pwd = request.recv(1024).decode('utf-8')
    usr_md5 = hashlib.md5()
    usr_md5.update(user_pwd.encode('utf-8'))
    md_pwd = usr_md5.hexdigest()
    if user_id in user.keys() and md_pwd == user[user_id]:
        request.send('登录成功'.encode('utf-8'))
    else:
        request.send('用户名或秘密错误，请重新输入'.encode('utf-8'))
    return user_id

def switch_usr_dir(user_id):
    #! 进入用户的目录
    os.chdir("../user_dir/" + user_id)    


def login(request):
    user_data = get_user_data()
    user_id = verify(user_data, request)
    switch_usr_dir(user_id)

if __name__ == '__main__':
    #? 测试账户:jj 密码:iiikkk
    login()
    print(os.getcwd())