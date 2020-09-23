import socketserver
import socket

class Client:
    '''
    Used to find servers on the network
    - ping nodes
    - issue jobs
    - retrieve results
    '''
    def __init__(self):
        self.port = 666
        self.result_dir = "results/"
        self.socket = MySocket()#socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        self.servers = []
        pass

    def ping(self, host):
        #s.connect((host, self.port))
        self.socket.connect('127.0.0.1', self.port)
        self.socket.send(b'Hello World')
        pass

    def remote_start_job(self):
        pass

    def remote_get_results(self):
        pass

class Node:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 666
        self.files = "data/"
        self.server = socketserver.TCPServer((self.host, self.port), MyTCPHandler)
        pass

    def start(self):
        self.log(f"STRTING SERVER ON\tHOST:{self.host}\tPORT:{self.port}")
        self.server.serve_forever()
        pass 

    def stop(self):
        pass

    def log(self, content):
        print(content)

    def job(self):
        '''
        What work to do
        maybe leave blank and override?
        '''
        pass

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())

class MySocket:
    """demonstration class only
      - coded for clarity, not efficiency
    """

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))

    def send(self, msg):
        totalsent = 0
        MSGLEN = len(msg)
        while totalsent < MSGLEN:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def receive(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)

if __name__ == '__main__':
    import time
    import sys

    if sys.argv[1] == "-c":
        c = Client()
        time.sleep(3)
        c.ping('127.0.0.1')
    elif sys.argv[1] == "-n":
        n = Node()
        n.start()