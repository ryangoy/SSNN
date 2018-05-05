import numpy as np
from pyntcloud.io import read_ply, write_ply
import pandas as pd
import json
import time
import zipfile
import collections
import zipfile

from os import listdir, remove, rmdir, rename, makedirs
from os.path import isdir, join, exists
import sys

def triangle_area_multi(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)


def create_reverse_map(segment_ids):
    max_seg_id = max(segment_ids)
    reverse_map = dict(zip(list(range(max_seg_id+1)), [[] for _ in range(max_seg_id+1)]))

    for i in range(len(segment_ids)):
        reverse_map[segment_ids[i]].append(i)

    return reverse_map

def get_label_vertices(seggroups, fgroups, vertices, room_points_xyz):
    
    reverse_fgroups = create_reverse_map(fgroups)

    bboxes = []
    labels = []


    for seggroup in seggroups:
        segments = []
        for i in seggroup["segments"]:
            segments += reverse_fgroups[i]

        v1_test = vertices["v1"][segments]
        v2_test = vertices["v2"][segments]
        v3_test = vertices["v3"][segments]

        v1_xyz = room_points_xyz[v1_test]
        v2_xyz = room_points_xyz[v2_test]
        v3_xyz = room_points_xyz[v3_test]

        bboxes.append([min(v1_xyz[:,0]), min(v1_xyz[:,1]), min(v1_xyz[:,2]), 
                       max(v1_xyz[:,0]), max(v1_xyz[:,1]), max(v1_xyz[:,2])])
        labels.append(seggroup["label"])

    return np.array(bboxes), np.array(labels)



# Code adapted from https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
def ply_to_xyz(room_fname, label_fname, fsegs_fname, point_density=20000):
    room = read_ply(room_fname)
    label = json.load(open(label_fname))
    fsegs = json.load(open(fsegs_fname))
    seggroups = label["segGroups"]
    fgroups = np.array(fsegs["segIndices"])



    # xyz points of each vertex
    points = room["points"]

    # id of xyz points that make up the triangle
    vertices = room["mesh"]

    room_points_xyz = points[["x", "y", "z"]].values
    room_points_rgb = points[["red", "green", "blue"]].values

    bboxes, labels= get_label_vertices(seggroups, fgroups, vertices, room_points_xyz)


    v1_xyz = room_points_xyz[room["mesh"]["v1"]]
    v2_xyz = room_points_xyz[room["mesh"]["v2"]]
    v3_xyz = room_points_xyz[room["mesh"]["v3"]]

    v1_rgb = room_points_rgb[room["mesh"]["v1"]]
    v2_rgb = room_points_rgb[room["mesh"]["v2"]]
    v3_rgb = room_points_rgb[room["mesh"]["v3"]]


    # areas = triangle_area_multi(v1_xyz, v2_xyz, v3_xyz)
    # probabilities = areas/areas.sum()

    # x_dim = max(v1_xyz[:,0]) - min(v1_xyz[:,0])
    # y_dim = max(v1_xyz[:,1]) - min(v1_xyz[:,1])
    # z_dim = max(v1_xyz[:,2]) - min(v1_xyz[:,2])

    # n = int(x_dim*y_dim*z_dim*point_density)

    # weighted_random_indices = np.random.choice(range(len(areas)), size=n, p=probabilities)

    # v1_xyz = v1_xyz[weighted_random_indices]
    # v2_xyz = v2_xyz[weighted_random_indices]
    # v3_xyz = v3_xyz[weighted_random_indices]

    # v1_rgb = v1_rgb[weighted_random_indices]
    # v2_rgb = v2_rgb[weighted_random_indices]
    # v3_rgb = v3_rgb[weighted_random_indices]

    # u = np.random.rand(n, 1)
    # v = np.random.rand(n, 1)
    # is_a_problem = u + v > 1

    # u[is_a_problem] = 1 - u[is_a_problem]
    # v[is_a_problem] = 1 - v[is_a_problem]

    # w = 1 - (u+v)

    result = pd.DataFrame()

    # result_xyz = (v1_xyz*u) + (v2_xyz*v) + (v3_xyz*w)
    # result_xyz = result_xyz.astype(np.float32)

    # result_rgb = (v1_rgb*u) + (v2_rgb*v) + (v3_rgb*w)
    # result_rgb = result_rgb.astype(np.float32)

    # result["x"] = result_xyz[:,0]
    # result["y"] = result_xyz[:,1]
    # result["z"] = result_xyz[:,2]

    result["x"] = v1_xyz[:,0]
    result["y"] = v1_xyz[:,1]
    result["z"] = v1_xyz[:,2]

    result["r"] = v1_rgb[:,0]
    result["g"] = v1_rgb[:,1]
    result["b"] = v1_rgb[:,2]



    # result["red"] = result_rgb[:,0]
    # result["green"] = result_rgb[:,1]
    # result["blue"] = result_rgb[:,2]

    return result, bboxes, labels


def main():
    start_t = time.time()
    room_fname = "scans/1LXtFkjw3qL/region_segmentations/region0.ply"
    label_fname = "scans/1LXtFkjw3qL/region_segmentations/region0.semseg.json"
    fsegs_fname = "scans/1LXtFkjw3qL/region_segmentations/region0.fsegs.json"
    result = ply_to_xyz(room_fname, label_fname, fsegs_fname)
    write_ply("test.ply", points=result)
    end_t = time.time()

    total_time = float(end_t-start_t)
    print("Took {} seconds".format(total_time))

def reorg_fstructure(path):
    already_processed = []
    start_t = time.time()
    num_areas = len(listdir(path))
    area_id = 1
    for i in listdir(path):
        if isdir(join(path,i)):
            print(i)
            try:
                zip_ref = zipfile.ZipFile(join(path,i,'region_segmentations.zip'),'r')
                zip_ref.extractall(join(path))
                zip_ref.close()
                remove(join(path,i,'region_segmentations.zip'))
            except:
                continue
            rs = join(path,i,"region_segmentations")
            if isdir(rs):
                ri = 0
                region_left = True
                processed_path = join(path,i,"processed_regions")
                if not isdir(processed_path):
                    makedirs(processed_path)
                while region_left:
                    if not exists(join(rs, "region{}.ply".format(str(ri)))) or i in already_processed:
                        region_left = False
                        print("Finished {}/{} Areas. Processed {} regions from area {}.".format(area_id, num_areas, ri, i))
                        continue

                    try:
                        room_fname = join(rs, "region{}.ply".format(str(ri)))
                        label_fname = join(rs, "region{}.semseg.json".format(str(ri)))
                        fsegs_fname = join(rs, "region{}.fsegs.json".format(str(ri)))
                        res, bboxes, labels = ply_to_xyz(room_fname, label_fname, fsegs_fname)

                        res_save = join(processed_path, "region{}.ply".format(str(ri)))
                        label_save = join(processed_path, "region{}_labels.npy".format(str(ri)))
                        bboxes_save = join(processed_path, "region{}_bboxes.npy".format(str(ri)))

                        write_ply(res_save, points=res)
                        np.save(label_save, labels)
                        np.save(bboxes_save, bboxes)
                    except:
                        print("Region {} for area {} threw an error and did not get processed.".format(ri, i))
                        pass


                    ri += 1
        area_id += 1
    end_t = time.time()
    print("Total time: {} seconds.".format(end_t-start_t))



                


                    

if __name__ == '__main__':
    reorg_fstructure(sys.argv[1])
