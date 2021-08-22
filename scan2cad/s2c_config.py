import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'models/retrieval'))

from s2c_map import ID2NAME, NAME2CLASS

class Scan2CADDatasetConfig(object):
    def __init__(self):
        self.num_class = 35
        self.num_heading_bin = 12

        # Top 20 categories
        self.ShapenetIDToName = ID2NAME
        self.ShapenetNameToClass = NAME2CLASS
        self.ClassToName = {self.ShapenetNameToClass[t]:t for t in self.ShapenetNameToClass}

    def ShapenetIDtoClass(self, cat_id):
        assert(cat_id in self.ShapenetIDToName)
        cat_name = self.ShapenetIDToName[cat_id]
        cat_cls = self.ShapenetNameToClass[cat_name]
        return cat_cls

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle