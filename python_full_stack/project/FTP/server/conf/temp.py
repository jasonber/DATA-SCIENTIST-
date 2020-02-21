import json
import os

# os.chdir(os.getcwd() + r'/project/chat/server/conf')
# with open('settings.json', encoding='utf-8') as f:
#     data = json.loads(f.read())
#
# ip_port = (data['IP'], data['PORT'])
#
# print('ip_port:{}\ntype:{}'.format(ip_port, type(ip_port)))


print(__file__)
file_path = "\\".join(__file__.split(r'/')[:-1])
print(file_path)
os.chdir(file_path)
print(os.getcwd())