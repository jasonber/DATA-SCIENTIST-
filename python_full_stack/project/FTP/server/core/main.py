import socketserver
import json
import os
file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
# import sys
# sys.path.append(file_path)
from server_act import Action
import socketserver


HOME_OPT = {'登录': 'login', '注册': 'regist'}

def run():
    #? 获取配置文件
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line

    setting_dic = json.loads(setting_json)   
    IP_PORT = (setting_dic['IP'], setting_dic['PORT'])


    class MyServer(socketserver.BaseRequestHandler):
        def handle(self):
            act = Action(self.request)
            # todo 主页
            self.request.send("欢迎光临".encode('utf-8'))
            flag = self.request.recv(4).decode('utf-8')
            home_flag = ['登录', '注册']
            if hasattr(Action, HOME_OPT[home_flag[flag-1]]):
                getattr(Action, HOME_OPT[home_flag[flag-1]])()
            #? 与客户端交互
            while 1:
                act.run_cmd()
    
    #? 创建服务器
    server = socketserver.ThreadingTCPServer(IP_PORT, MyServer)
    
    #? 启动服务器
    print("Server is working......")
    server.serve_forever()
    
    
if __name__ == "__main__":
    run()