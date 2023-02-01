import glob
import os
import re

p = 'data/images/c_this_is_a_cut1234_0.jpg'

print(re.sub('\.[^/.]+$', '', os.path.basename(p).replace('c_','t_', 1)) + '.txt')

