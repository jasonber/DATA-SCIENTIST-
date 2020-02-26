import socketserver
import json
import os
file_path = '/'.join(__file__.split('/')[:-1])
os.chdir(file_path)
# import sys
# sys.path.append(file_path)
import server_cls
import socketserver

def run():
    #? 获取配置文件
    setting_json = ""
    with open('../conf/settings.json', 'r', encoding='utf-8') as f:
        for line in f.read():
            setting_json += line

    setting_dic = json.loads(setting_json)   
    IP_PORT = (setting_dic['IP'], setting_dic['PORT'])

    #? 创建服务器
    server = socketserver.ThreadingTCPServer(IP_PORT, server_cls.Server)
    
    #? 启动服务器
    print("Server is working......")
    server.serve_forever()
    
    
if __name__ == "__main__":
    run()