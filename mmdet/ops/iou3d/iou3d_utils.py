import torch
import mmdet.ops.iou3d.iou3d_cuda as iou3d_cuda
import math
from scipy.spatial import Delaunay
import scipy

def limit_period(val, offset=0.5, period=math.pi):
    return val - torch.floor(val / period + offset) * period

def boxes3d_to_near_torch(boxes3d):
    rboxes = boxes3d[:, [0, 1, 3, 4, 6]]
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        boxes_near: [N, 4(xmin, ymin, xmax, ymax)] nearest boxes
    """
    rots = rboxes[..., -1]
    rots_0_pi_div_2 = torch.abs(limit_period(rots, 0.5, math.pi))
    cond = (rots_0_pi_div_2 > math.pi / 4)[..., None]
    boxes_center = torch.where(cond, rboxes[:, [0, 1, 3, 2]], rboxes[:, :4])
    boxes_near = torch.cat([boxes_center[:, :2] - boxes_center[:, 2:] / 2, \
                        boxes_center[:, :2] + boxes_center[:, 2:] / 2], dim=-1)
    return boxes_near

def boxes_iou(bboxes1, bboxes2, mode='iou', eps=0.0):
    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, cols)

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]
    wh = (rb - lt + eps).clamp(min=0)  # [rows, cols, 2]
    overlap = wh[:, :, 0] * wh[:, :, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + eps) * (
        bboxes1[:, 3] - bboxes1[:, 1] + eps)
    if mode == 'iou':
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + eps) * (
            bboxes2[:, 3] - bboxes2[:, 1] + eps)
        ious = overlap / (area1[:, None] + area2 - overlap)
    else:
        ious = overlap / (area1[:, None])
    return ious

# def boxes3d_to_bev_torch(boxes3d):
#     """
#     :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in lidar
#     :return:
#         boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
#     """
#     boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

#     cu, cv = boxes3d[:, 0], boxes3d[:, 1]
#     half_l, half_w = boxes3d[:, 3] / 2, boxes3d[:, 4] / 2
#     boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
#     boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
#     boxes_bev[:, 4] = boxes3d[:, 6]
#     return boxes_bev

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

def boxes3d_to_corners3d_lidar_torch(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry in KITTI dataset
    :param z_bottom: whether z is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6]
    ry = boxes3d[:, 6:7]

    zeros = torch.cuda.FloatTensor(boxes_num, 1).fill_(0)
    ones = torch.cuda.FloatTensor(boxes_num, 1).fill_(1)
    x_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim=1)  # (N, 8)
    y_corners = torch.cat([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dim=1)  # (N, 8)
    if bottom_center:
        z_corners = torch.cat([zeros, zeros, zeros, zeros, h, h, h, h], dim=1)  # (N, 8)
    else:
        z_corners = torch.cat([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dim=1)  # (N, 8)
    temp_corners = torch.cat((
        x_corners.unsqueeze(dim=2), y_corners.unsqueeze(dim=2), z_corners.unsqueeze(dim=2)
    ), dim=2)  # (N, 8, 3)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, -sina, zeros], dim=1)  # (N, 3)
    raw_2 = torch.cat([sina,  cosa, zeros], dim=1)  # (N, 3)
    raw_3 = torch.cat([zeros, zeros, ones], dim=1)  # (N, 3)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1), raw_3.unsqueeze(dim=1)), dim=1)  # (N, 3, 3)

    rotated_corners = torch.matmul(temp_corners, R)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]
    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.view(-1, 1) + x_corners.view(-1, 8)
    y = y_loc.view(-1, 1) + y_corners.view(-1, 8)
    z = z_loc.view(-1, 1) + z_corners.view(-1, 8)
    corners = torch.cat((x.view(-1, 8, 1), y.view(-1, 8, 1), z.view(-1, 8, 1)), dim=2)

    return corners


def boxes3d_to_corners3d_lidar(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry in KITTI dataset
    :param z_bottom: whether z is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    y_corners = np.array([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dtype=np.float32).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                         [np.sin(ry), np.cos(ry),  zeros],
                         [zeros,      zeros,        ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_to_corners3d_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 0:4] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros,      ones,        zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 4] / 2, boxes3d[:, 3] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev

def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a_bev.shape[0], boxes_b_bev.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, w, l, h, ry] in lidar coords
    :param boxes_b: (M, 7) [x, y, z, w, l, h, ry] in lidar coords
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_() # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5]).view(-1, 1)
    boxes_a_height_min = boxes_a[:, 2].view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5]).view(1, -1)
    boxes_b_height_min = boxes_b[:, 2].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d



def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()

def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()

class RotateIou2dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2):
        return boxes_iou_bev(boxes1, boxes2)

class RotateIou3dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2):
        return boxes_iou3d_gpu(boxes1, boxes2)


class NearestIouSimilarity(object):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """

    def __call__(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """

        boxes1_near = boxes3d_to_near_torch(boxes1)
        boxes2_near = boxes3d_to_near_torch(boxes2)
        return boxes_iou(boxes1_near, boxes2_near)

if __name__ == '__main__':
    pass