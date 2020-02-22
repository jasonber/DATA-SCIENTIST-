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
            command = self.request.recv(1024).decode('utf-8')
            main.trans_data(command, self.request)

    server = socketserver.ThreadingTCPServer(IP_PORT, My_server)
    server.serve_forever()



if __name__ == "__main__":
    main()
