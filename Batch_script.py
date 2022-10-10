import bpy
import os
import string
import random
import numpy as np
import pandas as pd

# Path to Motion files

path_to_bvh = 'C:\\path\\to\\bvh\\'
np.set_printoptions(precision=4, suppress=True)
file_list = sorted(os.listdir(path_to_bvh))

autodesk_list = [item for item in file_list if item.endswith('.bvh')]
datamotion = {item: pd.DataFrame() for item in autodesk_list}
dataiter = []

for item in autodesk_list:
    bpy.ops.object.empty_add(type='ARROWS', view_align=False, location=(0, 0, 0), layers=(True, False, False, False,
                                                                                          False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    bpy.data.objects['Empty'].name = "Empty.R"
    bpy.ops.object.empty_add(type='ARROWS', view_align=False, location=(0, 0, 0), layers=(True, False, False, False,
                                                                                          False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    bpy.data.objects['Empty'].name = "Empty.L"
    bpy.ops.object.empty_add(type='ARROWS', view_align=False, location=(0, 0, 0), layers=(True, False, False, False,
                                                                                          False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    bpy.data.objects['Empty'].name = "Empty.M"

    path_to_files = os.path.join(path_to_bvh, item)
    bpy.ops.import_anim.bvh(filepath=path_to_files, use_fps_scale=True,  axis_forward='Y', axis_up='Z',
                            update_scene_fps=True, update_scene_duration=True)
    it = item.replace('.bvh', '')
    endframe = int(bpy.data.objects[it].animation_data.action.frame_range.y)
    framerange = range(0, endframe)
    bpy.context.scene.frame_end = endframe+1
    bpy.data.objects['Empty.R'].parent = bpy.data.objects[it]
    bpy.data.objects['Empty.L'].parent = bpy.data.objects[it]
    bpy.data.objects['Empty.M'].parent = bpy.data.objects[it]
    bpy.data.objects['Empty.R'].parent_type = 'BONE'
    bpy.data.objects['Empty.L'].parent_type = 'BONE'
    bpy.data.objects['Empty.M'].parent_type = 'BONE'
    bpy.data.objects['Empty.R'].parent_bone = "rHand"
    bpy.data.objects['Empty.L'].parent_bone = "lHand"
    bpy.data.objects['Empty.M'].parent_bone = "hip"
    R_vector = (-6.0, 6.0, 1.0)
    L_vector = (6.0, 6.0, 1.0)
    bpy.data.objects['Empty.R'].matrix_local.translation.xyz = R_vector
    bpy.data.objects['Empty.L'].matrix_local.translation.xyz = L_vector
    M_root = bpy.data.objects['Empty.M'].matrix_world
    def add_noise(a):
        return(a+loc_noise)
    #dataiter = {iter: pd.DataFrame() for iter in range(0, 20)}
    for i in range(0, 20):
        tr_noise = random.uniform(-15, 15) * np.random.rand(1, 3)
        #rot_noise = np.random.rand(1, 3)
        dataframe = {frame: pd.DataFrame() for frame in framerange}
        for f in framerange:
            loc_noise = random.uniform(-1.5, 1.5) * np.random.rand(1, 3)
            bpy.context.scene.frame_set(f)
            #Rh_rot = np.array(bpy.data.objects['Empty.R'].matrix_world.to_euler()) + rot_noise
            #Lh_rot = np.array(bpy.data.objects['Empty.L'].matrix_world.to_euler()) + rot_noise
            M_root_inv = M_root.inverted()
            Rh_mat =bpy.data.objects['Empty.R'].matrix_world
            Lh_mat =bpy.data.objects['Empty.L'].matrix_world
            Rh_rel = M_root_inv * Rh_mat 
            Lh_rel = M_root_inv * Lh_mat 
            Rrh_loc= np.apply_along_axis(add_noise, 0, Rh_rel.translation.xyz)
            Lrh_loc= np.apply_along_axis(add_noise, 0, Lh_rel.translation.xyz)
            Rh_loc = Rrh_loc + tr_noise
            Lh_loc = Lrh_loc + tr_noise
            dataframe[f] = np.append(Rh_loc, Lh_loc).ravel()
        data1 = pd.DataFrame.from_dict(dataframe, orient='index')
        dataiter.append(data1)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # print(dataframe)
# print(datamotion)
data = pd.concat(dataiter, axis=0)
data.columns = ['X', 'Y', 'Z', 'X', 'Y', 'Z']

 data.to_csv('C:\\Users\\AP38100\\molab\\CSV\\08_03.csv')
