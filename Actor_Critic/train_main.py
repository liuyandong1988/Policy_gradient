#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@File    : train_main.py
@Time    : 2020/3/13 23:43
@Author  : Yandong
@Function : 
"""
import  a2c

def main():
    """
    Training AC model
    :return:
    """
    model = a2c.AC()
    history = model.train(200)
    model.save_history(history, 'ac_sparse.csv')
    print('Finish ...')



if __name__ == '__main__':
    main()