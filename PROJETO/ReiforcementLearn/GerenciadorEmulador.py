# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:06:14 2020

@author: mateu
"""

import subprocess
import socket
import threading
import time

conn = None
callbacksThread = None

class FceuxManager():
    def __init__(self):
        self.fceuxPath = '../AMBIENTE TESTE/FCEUX//fceux.exe'
        self.gamePath = '../AMBIENTE TESTE/games/Castlevania.nes'
        self.scriptPath = '../AMBIENTE TESTE/scripts_lua/CastlevaniaRamWatch.lua'
        self.scriptArgument = '-lua: '+self.scriptPath
        
    def sendComand(self, command):
        
    
    def start(self):
       subprocess.call([self.fceuxPath, self.gamePath,], shell=True, start_new_session=True),
       self.startServer()
    
    def startServer(self,):
        print('Start py-code')
        global conn, callbacksThread
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind(("127.0.0.1", 12345))
        s.listen(1)
        s.setblocking(1)
        print("Waiting connection from emulator...")
        conn, addr = s.accept()
        conn.setblocking(1)
        conn.settimeout(0.001)
        print("Connected: ", conn)
        callbacksThread = self.startCallbacksThread()
        print("Thread for listening callbacks from emulator started")
       
    def startCallbacksThread(self,):
        t = threading.Thread(target = callbacksThread)
        t.daemon = True
        t.start()
        return t
       
if __name__ == '__main__':
    fceuxManager = FceuxManager()
    fceuxManager.start()
    while True:
        time.sleep(60)