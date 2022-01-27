import torch
import torchvision

def draw_plot_on_tb(tb_writer, step, rgb_img, grasp_q_img, pred_q_img, depth_img=None, 
                    grasp_angle_img=None, grasp_width_img=None, pred_angle_img=None, pred_width_img=None):
    n_b, n_c, w, h = rgb_img.size()
    flat_rgb = rgb_img.view(n_b, -1)
    zero2one = flat_rgb - flat_rgb.min(1,keepdim=True)[0]
    view_rgb = zero2one.view(n_b, n_c, w, h)
    tb_writer.add_image('gt/rgb', torchvision.utils.make_grid(view_rgb), step)
    if depth_img is not None:
        view_depth = (depth_img + 1) * 0.5
        tb_writer.add_image('gt/depth', torchvision.utils.make_grid(view_depth), step)
    tb_writer.add_image('gt/quality', torchvision.utils.make_grid(grasp_q_img), step)
    tb_writer.add_image('pred/quality', torchvision.utils.make_grid(pred_q_img), step)