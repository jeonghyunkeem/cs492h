# Jeonghyun Kim, KAIST 

""" 
    Mapping dictionary of ShapeNetCore category id to Scan2CAD category name

    ALL_CATEGORY: a dictonary of {cat_id : cat_name} used for Scan2CAD annotation
    NAME2CLASS: a dictionary of {cat_name: cat_class}
            0 ~ 7   : cared cat_ids
            8 ~ 35  : others

    * CARED_LIST: a list of cared cat_ids in Scan2CAD benchmark 
"""

ID2NAME = {
    # 35 categories / Mapping from ID to NAME
    '02747177': 'trash bin',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub', 
    '02818832': 'bed',
    '02828884': 'bench',
    '02871439': 'bookshelf',
    '02876657': 'bottle', 
    '02880940': 'bowl',
    '02933112': 'cabinet',
    '02946921': 'can', 
    '02954340': 'cap',
    '03001627': 'chair',
    '03046257': 'clock',
    '03085013': 'keyboard',
    '03207941': 'dishwasher',
    '03211117': 'display',
    '03325088': 'faucet',
    '03337140': 'file cabinet', 
    '03467517': 'guitar', 
    '03593526': 'jar', 
    '03636649': 'lamp',
    '03642806': 'laptop',
    '03691459': 'speaker',
    '03761084': 'microwave',
    '03790512': 'motorcycle', 
    '03928116': 'piano', 
    '03938244': 'pillow', 
    '03991062': 'pot', 
    '04004475': 'printer',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04379243': 'table',
    '04401088': 'telephone', 
    '04554684': 'washer'
} 

NAME2CLASS = {
    # ===== CARED =====
    # Top 8 categories of Scan2CAD
    'bathtub': 0,
    'bookshelf': 1,
    'cabinet': 2, 
    'chair': 3, 
    'display': 4,
    'sofa': 5,
    'table': 6, 
    'trash bin': 7, 
    # ===== OTHERS =====    
    'bag': 8,
    'basket': 9,
    'bed': 10,
    'bench': 11,
    'bottle': 12, 
    'bowl': 13,
    'can': 14, 
    'cap': 15,
    'clock': 16,
    'keyboard': 17,
    'dishwasher': 18,
    'faucet': 19,
    'file cabinet': 20, 
    'guitar': 21, 
    'jar': 22, 
    'lamp': 23,
    'laptop': 24,
    'speaker': 25,
    'microwave': 26,
    'motorcycle': 27, 
    'piano': 28, 
    'pillow': 29, 
    'pot': 30, 
    'printer': 31,
    'stove': 32,
    'telephone': 33, 
    'washer': 34
}

# CARED_LIST = ['02747177', '02808440', '02871439', '02933112', '03001627', '03211117', '04256520', '04379243']