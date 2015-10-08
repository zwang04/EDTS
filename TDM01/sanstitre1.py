# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:00:52 2015

@author: zwang04
"""

import numpy as np
import scipy as sc
import matplotlib as ma
# a=np.zeros(3,4)
# b=np.random.randn(4,5)
 
x1=np.linspace(-2*pi,2*pi,100)
x2=np.linspace(-2*pi,6*pi,100)

y1=-np.sin(x1/2)
y2=np.sin(x2)
plot(x1,y1)
plot((x2-pi)/1.5,y2)

plt.fill(x1, y1, 'b', (x2-pi)/1.5, y2, 'r', alpha=0.3)
