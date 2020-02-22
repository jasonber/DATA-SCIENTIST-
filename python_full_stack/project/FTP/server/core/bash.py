import subprocess
import os
file_path ="/".join(__file__.split('/')[:-2])
usr_path = file_path + '/user_dir'
os.chdir(usr_path)
print(os.getcwd())


def bash(cmd):
    proc = subprocess.Popen(cmd, 
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            shell=True)

    res = proc.stdout.read().decode('utf-8')
    err = proc.stderr.read().decode('utf-8')

    if err:print(err)
    else: print(res)

def bash_cmd(command):
    try:
        cmd, directory = command.split(" ")
    except ValueError:
        print('''命令错误\n请按照"命令 目录"的格式输入命令''')
    try:
        if cmd == "查看":
            cmd_lst = "ls" + " " + directory
            bash(cmd_lst)
        elif cmd == "新建":
            cmd_lst = 'mkdir -p' + " " + directory
            bash(cmd_lst)
        elif cmd == "删除":
            cmd_lst = "rm" + " " + directory
            bash(cmd_lst)
        elif cmd == "切换":
            cmd_lst = "cd" + " " + directory
            os.chdir(directory)
            print("当前目录：{}".format(os.getcwd()))
    except UnboundLocalError: 
        pass

def run():
    command = input('请输入命令:') #? 有四个命令：查看、新建、删除、切换
    bash_cmd(command)

if __name__ == "__main__":
    while 1:
       run()
