# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:33:08 2020

@author: mateu
"""
import socket
import time
import json
import pickle
import matplotlib.pyplot as plt



class Server():
    def __init__(self):
        self.conn = self.waitForConnection()
        self.buff_size = 1024
    
    def waitForConnection(self,):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind(("127.0.0.1", 12345))
        s.listen(1)
        s.setblocking(1)
        print("Waiting connection from emulator...")
        conn, addr = s.accept()
        conn.setblocking(4)
        conn.settimeout(30)
        print("Connected: ", conn)
        return conn
        #callbacksThread = startCallbacksThread()
        #print("Thread for listening callbacks from emulator started")
        
    def sendCommandAndReceiveOperation(self, message):
        self.conn.send(message.encode('utf-8'))
        data = self.recvall()
        #data = json.loads(data.decode())
        return data
        
    def receiveMessage(self,):
        data = None
        try:
            data = self.conn.recv(self.buff_size)
        except socket.timeout:
            print('Pass')
        return data

    def recvall(self,):
        data = b''
        while True:
            part = self.conn.recv(self.buff_size)
            data += part
            if len(part) < self.buff_size:
                # either 0 or end of data
                break
        return data        


    def closeServer(self,):
        self.conn.close()
    

if __name__ == "__main__":
    server = Server()
    #server.sendCommand('mensagem')
    while True:
        time.sleep(60)        