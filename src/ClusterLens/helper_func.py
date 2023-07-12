import numpy as np 
import sys

def check_sysargv(knob, fall_back, d_type, sysargv=None):
    if sysargv is None:
        sysargv = np.array(sys.argv); print(['{}'.format(i) for i in sysargv])
    smooth_info = np.array([knob in a for a in sysargv])
    if np.any(smooth_info): 
        smooth_info = np.array(sysargv)[smooth_info]
        smooth_info = smooth_info[0].split(knob)[-1]
        smooth = d_type(smooth_info)
    else:
        try:
            smooth = d_type(fall_back)
        except:
            return None
    return smooth 