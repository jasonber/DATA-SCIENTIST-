import json
import os 

os.chdir(os.getcwd() + r'/project/chat/server/conf')
with open('settings.json', encoding='utf-8') as f:
    data = json.loads(f.read())

ip_port = (data['IP'], data['PORT'])

print('ip_port:{}\ntype:{}'.format(ip_port, type(ip_port)))