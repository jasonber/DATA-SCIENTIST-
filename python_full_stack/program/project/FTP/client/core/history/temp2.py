import subprocess

cmd = input(">>>")

res = subprocess.Popen(cmd, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       stdin=subprocess.PIPE, 
                       shell=True)

print("err:",res.stderr.read())
print("out:",res.stdout.read())