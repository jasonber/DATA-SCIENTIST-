import socketserver
import json
import os
file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
import sys
sys.path.append(file_path)
import login
import regist 
import trans_data
import homepage
import server_cmd


def main():
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line

    setting_dic = json.loads(setting_json)
    IP_PORT = (setting_dic['IP'], setting_dic['PORT'])
    
    #? 创建并发服务器
    class My_server(socketserver.BaseRequestHandler):
        def handle(self):
            #? 提示服务启动
            print("Server is working.....")
            #? 进入主页
            homepage.homepage(self.request)
            #? 与客户端通讯
            while 1:
                #? 等待客户端的命令
                command = self.request.recv(1024).decode('utf-8')
                server_cmd.server_cmd(command, self.request)

    server = socketserver.ThreadingTCPServer(IP_PORT, My_server)
    server.serve_forever()



if __name__ == "__main__":
    main()
