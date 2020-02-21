import hashlib
import os

file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)

# usr_dic = {}
# with open(r'/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/user', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         usr_name, usr_pwd = line.split(" ")
#         usr_dic[usr_name] = usr_pwd

# #! 用户名重复验证
# new_name = input('请输入用户名:')
# while new_name in usr_dic.keys():
#     print("用户名重复")
#     new_name = input("请输入用户名:")

# #! 密码加密
# new_pwd = input('请输入密码:')
# new_md5 = hashlib.md5()
# new_md5.update(new_pwd.encode('utf-8'))
# md_pwd = new_md5.hexdigest()

# #! 用户名和密码存储
# new_usr = new_name + " " + md_pwd +'\n'
# with open(r'/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/user', 'a', encoding='utf-8', newline=None) as f:
#     f.write(new_usr)

# print("注册成功！")

def get_user_data():
    user_data = {}
    with open(r'../DB/user', 'r', encoding='utf-8', newline=None) as f:
        for line in f.readlines():
            user_id, user_pwd = line.split(" ")
            user_data[user_id] = user_pwd[:-1]
    return user_data

def regist(path):
    #! 用户名重复验证
    new_name = input('请输入用户名:')
    while new_name in usr_dic.keys():
        print("用户名重复")
        new_name = input("请输入用户名:")

    #! 密码加密
    new_pwd = input('请输入密码:')
    new_md5 = hashlib.md5()
    new_md5.update(new_pwd.encode('utf-8'))
    md_pwd = new_md5.hexdigest()

    #! 用户名和密码存储
    new_usr = new_name + " " + md_pwd +'\n'
    with open(r'../DB/user', 'a', encoding='utf-8', newline=None) as f:
        f.write(new_usr)

    print("注册成功！")