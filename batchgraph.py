# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:14:25 2016

@author: Carlos
"""

import matplotlib.pyplot as plt
import datetime as dt

fig = plt.figure()
plt.plot(range(10))
s_dt = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')
fig.savefig('temp'+s_dt+'.png', dpi=fig.dpi)