#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:47:55 2020

@author: thomas
"""

import cat_slice as cs
import numpy as np

a = cs.CatSlice(start = 0, stop=4, step=1, input_array=np.array([1, 2, 3]))
b = cs.CatSlice(start = 0, stop=5, step=1, input_array=np.array([4, 5, 6]))

c = a + b

print(a)
print(b)
print("Result = ",  c)