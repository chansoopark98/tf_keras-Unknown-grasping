import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results
from utils.dataset_processing.grasp import detect_grasps

import rospy
from sensor_msgs.msg import Image as smImage
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import cv2

import time

bridge = CvBridge()
logging.basicConfig(level=logging.INFO)

hex_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
rgb_color_list = [tuple(int(ci[i:i+2], 16) for i in (1, 3, 5)) for ci in hex_color_list]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--networks', nargs='+', type=str, 
                        default='/root/plaif_vision/grasp_data/epoch_100_iou_0.63_0.65',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

class GraspNetworks:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, nets=None):
        if nets:
            self.nets = nets
        else:
            self.nets=[]

    def __getitem__(self, item):
        return self.nets[item]

    def __iter__(self):
        return self.nets.__iter__()
    
    def set_publisher(self):
        for id, network in enumerate(self.nets):
            setattr(network, 'rgb_pub', rospy.Publisher(f'net{id}_grasp', smImage))
            setattr(network, 'q_pub', rospy.Publisher(f'net{id}_quality', smImage))
            setattr(network, 'a_pub', rospy.Publisher(f'net{id}_angle', smImage))
            setattr(network, 'w_pub', rospy.Publisher(f'net{id}_width', smImage))
    
    def predict_all(self, x, rgb):
        xc = x.to(device)

        for id, network in enumerate(self.nets):
            draw_rgb = rgb.copy()
            with torch.no_grad():
                s_time = time.time()
                pred = network.net.predict(xc)
                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                gs = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=args.n_grasps)
                print(f'net{id:02} - grasps prediction time: {time.time()-s_time}')
            for gi, g in enumerate(gs):
                rect = ((rgb.shape[1]/2 - 336 + g.center[1].item(), rgb.shape[0]/2 - 336 + g.center[0].item()), (g.length, g.width), -g.angle*180/3.141592)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(draw_rgb, [box], 0, rgb_color_list[gi], 2)
            cv2.rectangle(draw_rgb,(int(rgb.shape[1]/2 - 336), int(rgb.shape[0]/2 - 336)), (int(rgb.shape[1]/2 + 336), int(rgb.shape[0]/2 + 336)), (255,255,255), thickness=2)

            rgb_msg = bridge.cv2_to_imgmsg(draw_rgb, "rgb8")
            q_img = bridge.cv2_to_imgmsg(q_img)
            width_img = bridge.cv2_to_imgmsg(width_img)
            ang_img = bridge.cv2_to_imgmsg(ang_img)

            network.rgb_pub.publish(rgb_msg)
            network.q_pub.publish(q_img)
            network.w_pub.publish(width_img)
            network.a_pub.publish(ang_img)
        print(f'========================================================================')

class GraspNetwork:
    def __init__(self, net):
        if net:
            self.net = torch.load(net)
        else:
            self.net = net
        self.rgb_pub = None
        self.q_pub = None
        self.a_pub = None
        self.w_pub = None

def get_data(cb_dep, cb_rgb):

    dep = bridge.imgmsg_to_cv2(cb_dep, desired_encoding='16UC1')
    rgb = bridge.imgmsg_to_cv2(cb_rgb, desired_encoding='rgb8')

    cam_data = CameraData(width=rgb.shape[1], height=rgb.shape[0], include_depth=args.use_depth, include_rgb=args.use_rgb, output_size=672)

    dep = dep[..., None]

    x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=dep)

    gr_nets.predict_all(x, rgb)

if __name__ == '__main__':
    args = parse_args()

    networks_list = args.networks
    
    networks = []
    for net_pth in networks_list:
        networks.append(GraspNetwork(net_pth))
    
    gr_nets = GraspNetworks(networks)
    gr_nets.set_publisher()

    # Get the compute device
    device = get_device(args.force_cpu)

    # Get data 
    rospy.init_node('Grasping')
    dep_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', smImage)
    rgb_sub = message_filters.Subscriber('/camera/color/image_rect_color', smImage)

    ts = message_filters.ApproximateTimeSynchronizer([dep_sub, rgb_sub], 10, 0.1)
    ts.registerCallback(get_data)

    rospy.spin()

