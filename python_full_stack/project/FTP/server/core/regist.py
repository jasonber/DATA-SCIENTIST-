import hashlib
import os

file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
import bash
# usr_dic = {}
# with open(r'/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/DB/user', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         usr_id, usr_pwd = line.split(" ")
#         usr_dic[usr_id] = usr_pwd

# #! 用户名重复验证
# new_id = input('请输入用户名:')
# while new_id in usr_dic.keys():
#     print("用户名重复")
#     new_id = input("请输入用户名:")

# #! 密码加密
# new_pwd = input('请输入密码:')
# new_md5 = hashlib.md5()
# new_md5.update(new_pwd.encode('utf-8'))
# md_pwd = new_md5.hexdigest()

# #! 用户名和密码存储
# new_usr = new_id + " " + md_pwd +'\n'
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

def create_usr(usr_dic, request):
    #! 用户名重复验证
    request.send("请输入用户名：".encode('utf-8'))
    new_id = request.recv(1024).decode('utf-8')
    while new_id in usr_dic.keys():
        request.send("用户名重复".encode('utf-8'))
        new_id = request.recv(1024).decode('utf-8')

    #! 密码加密
    request.send("请输入密码：".encode('utf-8'))
    new_pwd = request.recv(1024).decode('utf-8')
    new_md5 = hashlib.md5()
    new_md5.update(new_pwd.encode('utf-8'))
    md_pwd = new_md5.hexdigest()

    #! 用户名和密码存储
    new_usr = new_id + " " + md_pwd +'\n'
    with open(r'../DB/user', 'a', encoding='utf-8', newline=None) as f:
        f.write(new_usr)

    print("注册成功！")
    request.send("注册成功!".encode('utf-8'))

    #! 创建用户根目录
    cmd = 'mkdir ../user_dir/' + new_id
    bash.bash(cmd)
    print("创建用户目录")

def regist(request):
    create_usr(get_user_data(), request)


if __name__ == '__main__':
    regist()