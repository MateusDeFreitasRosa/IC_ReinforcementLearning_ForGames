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
import time

   
class Server():
    def __init__(self):
        self.buff_size = 1024
        self.conn = self.waitForConnection()


    def waitForConnection(self,):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.bind(("127.0.0.1", 12345))
            s.listen(1)
            s.setblocking(1)
            print("Waiting connection from emulator...")
            conn, addr = s.accept()
            conn.setblocking(4)
            conn.settimeout(1)
            print("Connected: ", conn)
            return conn
            #callbacksThread = startCallbacksThread()
            #print("Thread for listening callbacks from emulator started")
        except Exception as e:
            print('Error: {}'.format(e))
        
    def converToJson(self,string):
        a = string.decode('utf-8')
        return json.loads(a)
    
    def sendCommandAndReceiveOperation(self, message):
        try:
            self.conn.send(message.encode('utf-8'))
            data = self.recvall()
            #data = json.loads(data.decode())
            return self.converToJson(data)
        except Exception as e:
            print('Requisitando novamente {}'.format(str(type(e))))
            if (str(type(e)) != "<class 'socket.timeout'>"):
                self.closeServer()
                return
            
            time.sleep(.01)
            return self.sendCommandAndReceiveOperation(message)
            
        
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

    def step(self, action=None, grayscale=False, downsample=1, min_x=None, max_x=None, min_y=None, max_y=None):
        
        try:
            if action == None:
                raise Exception('action can\'t be None.')
            
            op = {}
            op['operation'] = 'nextStep'
            op['params'] = {}
            op['params']['screenshot_params'] = {}
            op['params']['action'] = action
            op['params']['screenshot_params']['grayscale'] = grayscale
            op['params']['screenshot_params']['down_sample'] = downsample
            if min_y != None:
                op['params']['screenshot_params']['len_min_y'] = min_y
            if max_y != None:
                op['params']['screenshot_params']['len_max_y'] = max_y
            if min_x != None:
                op['params']['screenshot_params']['len_min_x'] = min_x
            if max_x != None:
                op['params']['screenshot_params']['len_max_y'] = max_x
            
            return self.sendCommandAndReceiveOperation(json.dumps(op))
        except Exception as e:
            print(e)
        
    def reset(self, loadState=None, grayscale=False, downsample=1, min_x=None, max_x=None, min_y=None, max_y=None):
        try:
            
            op = {}
            op['operation'] = 'reset'
            op['params'] = {}
            op['params']['file_state'] = loadState
            op['params']['screenshot_params'] = {}
            op['params']['screenshot_params']['grayscale'] = grayscale
            op['params']['screenshot_params']['down_sample'] = downsample
            if min_y != None:
                op['params']['screenshot_params']['len_min_y'] = min_y
            if max_y != None:
                op['params']['screenshot_params']['len_max_y'] = max_y
            if min_x != None:
                op['params']['screenshot_params']['len_min_x'] = min_x
            if max_x != None:
                op['params']['screenshot_params']['len_max_y'] = max_x
                
            return self.sendCommandAndReceiveOperation(json.dumps(op))
        except Exception as e:
            print(e)            
        
    def registerMap(self, mapMemory):
        self.sendCommandAndReceiveOperation(json.dumps(
            {'operation': 'registerMap',
              'params': {
                  'tableMap': mapMemory,
            }
        }))
        
    def closeServer(self,):
        self.conn.shutdown(socket.SHUT_RDWR)
        self.conn.close()
    
  