import socketserver
import json

setting_json = ""
with open('/mnt/d/个人目标/my_git_hub/DATA-SCIENTIST-/python_full_stack/project/FTP/server/conf/settings.json', 'r', encoding='utf-8') as f:
    for line in f.read():
        setting_json += line

setting_dic = json.loads(setting_json)
IP_PORT = (setting_dic['IP'], setting_dic['PORT'])

class My_serever(socketserver.BaseRequestHandler):
    def handle(self):
        pass

server = socketserver.ThreadingTCPServer(IP_PORT, My_server)
server.serve_forever()