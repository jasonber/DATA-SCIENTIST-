import struct

def cmd_res(request):
    cmd_header = request.recv(4)
    cmd_size = struct.unpack('i', cmd_header)
    recv_size = 0
    res = ""
    while recv_size < cmd_size:
        data = request.recv(1024).decode('utf-8')
        res += data
        recv_size += len(data)
    print(res)
