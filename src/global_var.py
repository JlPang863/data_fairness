# -*- coding: utf-8 -*-

def init():  
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    # define
    _global_dict[key] = value

def get_value(key):
    # get global variable
    try:
        return _global_dict[key]
    except:
        print('Read'+key+'Failed\r\n')