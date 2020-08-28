#!/usr/bin/env python
'''
visualize point cloud from numpy array
TODO: change visualize code and save pcd files
author  : Yuqiong Li, Ruoyu Wang
created : 05/02/19  11:26 AM
'''
import numpy as np
from matplotlib import pyplot as plt
import utils
from open3d import *


def vis_points(points ):
    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(points)
    # vis.get_render_option().load_from_json('renderopt.json')
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-40)
    vis.run()
    # vis.capture_screen_image(fname)
    vis.destroy_window()


def main():
    # for i in range(400):
        # opath = "../viz/city/o{0}.npy".format(i)   # original data
    ppath = "../viz/arr_0.npy"  # original data
    # o = np.load(opath)   # original data
    p = np.load(ppath)   # predicted data
    for j in range(p.shape[0]):
        # pcd = PointCloud()
        # pcd.points = Vector3dVector(o[j])
        pcd1 = PointCloud()
        pcd1.points = Vector3dVector(p[j])
        # vis_points(pcd)
        vis_points(pcd1)
        input("Press Enter to continue...")
    # open3d.write_point_cloud("../../TestData/sync.ply", pcd)
    # pptk.viewer(o[0])
    # pptk.viewer(p[0])
    # utils.vis_pts(o[0], 'b', 'tab20')
    # utils.vis_pts(o[0], 'b', 'tab20')
    # plt.show()
    # utils.vis_pts(p[0], 'b', 'tab20')
    # plt.show()
    return


if __name__ == "__main__":
    main()
