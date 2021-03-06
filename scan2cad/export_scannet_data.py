import os, sys
import csv, json
import numpy as np
from plyfile import PlyData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
HOME_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(BASE_DIR)

TRAIN_DIR = os.path.join(HOME_DIR, "Dataset/ScanNet/scans/")
VAL_DIR = os.path.join(HOME_DIR, "Dataset/ScanNet/scans_test/")
OUTPUT_DIR = os.path.join(BASE_DIR, "./scan2cad_data")

from s2c_config import Scan2CADDatasetConfig
from s2c_dataset import Scan2CADDataset

META_DIR = os.path.join(BASE_DIR, 'scannet_meta')
LABEL_MAP_FILE = META_DIR + '/scannetv2-labels.combined.tsv'
MAX_NUM_POINT = 40000

DC = Scan2CADDatasetConfig()
ClassToName = {int(DC.ShapenetNameToClass[k]):k for k in DC.ShapenetNameToClass}
print(ClassToName)

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

def label_mapping(filename, label_from='raw_category', label_to='ShapeNetCore55'):
    assert os.path.isfile(filename)
    mapping = {}

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row[label_to] == 'tv or monitor':
                mapping[row[label_from]] = 'display'
            elif row[label_to] == 'trash_bin':
                mapping[row[label_from]] = 'trash bin'
            elif row[label_to] == 'tub':
                mapping[row[label_from]] = 'bathtub'
            elif row[label_to] == 'tv or monitor':
                mapping[row[label_from]] = 'display'
            elif row[label_to] == 'computer keyboard':
                mapping[row[label_from]] = 'keyboard'
            elif row[label_to] == 'suitcase':
                mapping[row[label_from]] = 'bag'
            elif row[label_to] == 'washing_machine':
                mapping[row[label_from]] = 'washer'
            elif row[label_from] == 'file cabinet' or row[label_from] == 'file cabinets':
                mapping[row[label_from]] = 'file cabinet'
            elif row[label_to] not in DC.ShapenetNameToClass:
                mapping[row[label_from]] = 'other'
            else:
                mapping[row[label_from]] = row[label_to]
        
    return mapping

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    # print(filename)
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def export(scan_name, matrix, output_file, train=True):
    if train:
        SCANNET_DIR = TRAIN_DIR
    else:
        SCANNET_DIR = VAL_DIR

    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')

    # Vertices
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)
    axis_align_matrix = np.array(matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    label_map = label_mapping(LABEL_MAP_FILE)
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_str = label_map[label]
        assert DC.ShapenetNameToClass[label_str] is not None
        label_id = int(DC.ShapenetNameToClass[label_str])
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]

    object_status = {}
    instance_bboxes = np.zeros((num_instances,7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id-1,:] = bbox 

        # status
        if ClassToName[label_id] not in object_status:
            object_status[ClassToName[label_id]] = 1
        else:
            object_status[ClassToName[label_id]] += 1

    print(sorted(object_status.items(), key=lambda item: item[1], reverse=True))

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices   = mesh_vertices[choices, :]
        label_ids       = label_ids[choices]
        instance_ids    = instance_ids[choices]

    if output_file is not None:
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)

    return mesh_vertices, label_ids, instance_ids,\
        instance_bboxes, object_id_to_label_id

if __name__ == '__main__':
    condition = 'all' 
    DATASET = Scan2CADDataset(split_set=condition)
    for idx, scan_name in enumerate(DATASET.scan_names):
        id_scan, i_matrix = DATASET.get_alignment_matrix(idx)
        assert scan_name == id_scan
        print(idx, scan_name)
        output_file = os.path.join(OUTPUT_DIR, scan_name) 
        if os.path.isfile(output_file+'_bbox.npy'):
            print('File already exists. skipping.')
            print('-'*20+'done')
            continue
        export(scan_name=scan_name, matrix=i_matrix, output_file=output_file)
        

