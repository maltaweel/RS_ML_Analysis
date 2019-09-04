'''
Created on Sep 4, 2019

@author: mark
'''

import numpy as np
from pointpats import PointPattern


points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21],
          [9.47, 31.02],  [30.78, 60.10], [75.21, 58.93],
          [79.26,  7.68], [8.23, 39.93],  [98.73, 77.17],
          [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]
p1 = PointPattern(points)

p1.mbb

p1.summary()

type(p1.points)
np.asarray(p1.points)

p1.mbb

points = np.asarray(points)
points

p1_np = PointPattern(points)
p1_np.summary()






