import socket
import os

if __name__ == '__main__':
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 8039))
    instr = b'from client'
    client.send(instr)
    print(client.recv(1024))
    client.close()
