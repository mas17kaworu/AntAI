import SocketServer
from sandbox import get_sandbox
import time
import os
import sys

global bots
global turn
global loadtime
global turntime
global count
global current_bot
bots = []
turn = 0
loadtime = 3
turntime = 1
count = 0

class MyServer(SocketServer.BaseRequestHandler):
    def handle(self):
        global bots
        global turn
        global loadtime
        global turntime
        global count
        global current_bot
        global reveiveMsg
        print("Socket server is ready")
        while(True):
            reveiveMsg = ""
            while(True):
                revLine = self.request.recv(1024).strip()
                if revLine is None:
                    continue
                reveiveMsg = reveiveMsg + revLine.decode() + "\n"
                if 'go' in reveiveMsg.lower() or 'ready' in reveiveMsg.lower() or 'end' in reveiveMsg.lower():
                    break

            current_bot.write(reveiveMsg)
            start_time = time.time()
            bot_flag = True
            if turn == 0:
                time_limit = loadtime
            else:
                time_limit = turntime

            sendLine = ""
            while (bot_flag and time.time() - start_time < time_limit):
                time.sleep(0.01)
                line = current_bot.read_line()
                if line is None:
                    # stil waiting for more data
                    continue
                line = line.strip()
                if line.lower() == 'go':
                    bot_flag = False
                    turn = turn+1
                    count = count+1
                line = line+"\n"
                sendLine = sendLine + line

            self.request.sendall(bytes(sendLine))

     
def main(argv):
    global current_bot
    Host, Port = "127.0.0.1", 18889

    Host = argv[0]
    server = SocketServer.ThreadingTCPServer((Host,Port),MyServer)
    bot_cwd = os.getcwd() 
    bot_cmd = argv[1] 
    sandbox = get_sandbox(bot_cwd,False)
    sandbox.start(bot_cmd)
    bots.append(sandbox)
    
    current_bot = sandbox
    server.serve_forever()


if __name__=="__main__":
    sys.exit(main(sys.argv[1:]))
    