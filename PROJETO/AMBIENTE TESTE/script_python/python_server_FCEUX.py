# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:33:08 2020

@author: mateu
"""
import socket
import time
import json
import pickle



class Server():
    def __init__(self):
        self.conn = self.waitForConnection()
        self.buff_size = 1024*1024
    
    def waitForConnection(self,):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind(("127.0.0.1", 12345))
        s.listen(1)
        s.setblocking(1)
        print("Waiting connection from emulator...")
        conn, addr = s.accept()
        conn.setblocking(1)
        conn.settimeout(1)
        print("Connected: ", conn)
        return conn
        #callbacksThread = startCallbacksThread()
        #print("Thread for listening callbacks from emulator started")
        
    def sendCommandAndReceiveOperation(self, message):
        print('SendMessage {}'.format(message))
        #j = json.dumps(message)
        self.conn.send(message.encode('utf-8'))
        data = self.receiveMessage()
        data = json.loads(data.decode())
        print('Type: {}'.format(type(data)))
        #print('INFO[0][239]: {}'.format(data[0][238]))
        print('Data: {}'.format(data))
        
    
    def receiveMessage(self,):
        print('Stop here?')
        data = None
        try:
            data = self.conn.recv(self.buff_size)
        except socket.timeout:
            print('Pass?')
        return data

        
    def closeServer(self,):
        self.conn.close()
    

if __name__ == "__main__":
    server = Server()
    #server.sendCommand('mensagem')
    while True:
        time.sleep(60)        