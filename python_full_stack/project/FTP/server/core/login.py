import os
import hashlib

file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)

print("欢迎光临")
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
    with open(r'../server/DB/user', 'r', encoding='utf-8', newline=None) as f:
        for line in f.readlines():
            user_id, user_pwd = line.split(" ")
            user_data[user_id] = user_pwd[:-1]
    return user_data

def login(user_data):
    user_id = input('请输入用户名：')
    user_pwd = input('请输入密码：')
    usr_md5 = hashlib.md5()
    usr_md5.update(user_pwd.encode('utf-8'))
    md_pwd = usr_md5.hexdigest()
    if user_id in user.keys() and md_pwd == user[user_id]:
        print('登录成功')
    else:
        print('用户名或秘密错误，请重新输入')