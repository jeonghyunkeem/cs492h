import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class Scan2CADDatasetConfig(object):
    def __init__(self):
        self.num_class = 21 # 9
        self.num_heading_bin = 12

        # Top 20 categories
        self.ShapenetIDToName = {
                    '03001627': 'chair',
                    '04379243': 'table',
                    '02933112': 'cabinet',
                    '02747177': 'trash bin',
                    '02871439': 'bookshelf',
                    '03211117': 'display',
                    '04256520': 'sofa',
                    '02808440': 'bathtub',
                    "02818832": 'bed',
                    "03337140": 'file cabinet',
                    "02773838": 'bag',
                    "04004475": 'printer',
                    "04554684": 'washer',
                    "03636649": 'lamp',
                    "03761084": 'microwave',
                    "04330267": 'stove',
                    "02801938": 'basket',
                    "02828884": 'bench',
                    "03642806": 'laptop',
                    "03085013": 'keyboard'
                    }

        self.ShapenetNameToClass = {
            'chair': 0, 'table': 1, 'cabinet': 2, 'trash bin': 3, 'bookshelf': 4,'display': 5,'sofa': 6,
            'bathtub': 7,'bed': 8, 'file cabinet': 9, 'bag': 10, 'printer': 11, 'washer': 12, 'lamp': 13, 
            'microwave': 14, 'stove': 15, 'basket': 16, 'bench': 17, 'laptop': 18, 'keyboard': 19, 'other': 20
        }
        
        self.ClassToName = {self.ShapenetNameToClass[t]:t for t in self.ShapenetNameToClass}

    def ShapenetIDtoClass(self, id):
        if id in self.ShapenetIDToName:
            cad_category = self.ShapenetIDToName[id]
        else:
            cad_category = 'other'
        cad_class = self.ShapenetIDtoClass[cad_category]
        return cad_class

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