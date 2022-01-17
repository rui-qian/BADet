import torch
import torch.nn as nn
import logging
from mmcv.runner import load_checkpoint
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (assign_and_sample, bbox2roi, rbbox2roi, bbox2result, multi_apply, kitti_bbox2results,\
                        tensor2points, delta2rbbox3d, weighted_binary_cross_entropy)
import torch.nn.functional as F
from mmdet.models.gnns import BARefiner
from mmdet.models.gnns.pointnet2_msg import Pointnet2MSG
from mmdet.models.necks.cmn import nearest_neighbor_interpolate
class SingleStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.msg = Pointnet2MSG(input_channels=1)
        self.rgnn = BARefiner(state_dim=120, n_classes=1, n_iterations=3)
        self.fc = nn.Linear(512, 28, bias=False)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError
        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

        if extra_head is not None:
            self.extra_head = builder.build_single_stage_head(extra_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in [
                'voxels', 'num_points', 'image_points'
            ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in [
                'img_meta', 'gt_labels', 'gt_bboxes',
            ]:
                ret[key] = elems
            else:
                ret[key] = torch.stack(elems, dim=0)
        return ret
    
    def voxel_features_sampling(self, rois_centers,  points_mean, pointwise):
        p = nearest_neighbor_interpolate(rois_centers,  points_mean, pointwise)
        return p 

    def forward_train(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)
        ret = self.merge_second_batch(kwargs)
        ret['img'] = img
        losses = dict()
        vx = self.backbone(ret)
        (x, conv6), point_misc, points_mean, pointwise = self.neck(vx, ret['coordinates'], batch_size)
        aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
        losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            guided_anchors, new_scores = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], ret['gt_bboxes'], thr=0.1)
        else:
            raise NotImplementedError
        
        l_xyz, l_features = [],[]
        for bs in range(batch_size):
            mask = ret['coordinates'][...,0]==bs
            single_l_xyz, single_l_features = self.msg(vx[mask][None,...])
            single_l_xyz = F.pad(single_l_xyz, [1, 0, 0, 0],mode='constant',value=bs)
            l_xyz.append(single_l_xyz)
            l_features.append(single_l_features)
        l_xyz = torch.cat(l_xyz,0)
        l_features = torch.cat(l_features,0)
        
        # # bbox head forward and loss
        if self.extra_head:
            pixel_wise_features = self.extra_head(conv6, guided_anchors)
            rois_centers_list = []
            for bs_idx in range(batch_size):
                cur_roi = guided_anchors[bs_idx][:, 0:7].contiguous()
                cur_roi = F.pad(cur_roi, [1, 0],mode='constant',value=bs_idx)
                rois_centers_list.append(cur_roi[:,:4])
            rois_centers = torch.cat(rois_centers_list, dim=0) 
            voxel_wise_features = self.voxel_features_sampling(rois_centers,  points_mean, pointwise)

            l_features = self.fc(l_features)
            point_wise_features = nearest_neighbor_interpolate(rois_centers, l_xyz, l_features)
            ret['node_features'] = torch.cat((pixel_wise_features, voxel_wise_features, point_wise_features),-1)
            ret['node_pos'] = rois_centers
            rpn_outs, state = self.rgnn(ret)
            refine_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], guided_anchors, ret['anchors_mask'], self.train_cfg.extra)
            refine_losses = self.extra_head.loss(*refine_loss_inputs)
            losses.update(refine_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)
        ret = self.merge_second_batch(kwargs)
        ret['img'] = img
        vx = self.backbone(ret)
        (x, conv6), point_misc, points_mean, pointwise = self.neck(vx, ret['coordinates'], batch_size)
        rpn_outs = self.rpn_head.forward(x)

        guided_anchors,bbox_score = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                                       None, thr=0.1)
        l_xyz, l_features = [],[]
        for bs in range(batch_size):
            mask = ret['coordinates'][...,0]==bs
            single_l_xyz, single_l_features = self.msg(vx[mask][None,...])
            single_l_xyz = F.pad(single_l_xyz, [1, 0, 0, 0],mode='constant',value=bs)
            l_xyz.append(single_l_xyz)
            l_features.append(single_l_features)
        l_xyz = torch.cat(l_xyz,0)
        l_features = torch.cat(l_features,0)

        bbox_score, pixel_wise_features = self.extra_head(conv6, guided_anchors,is_test=True)
        if len(bbox_score)==0:
            pixel_wise_features = torch.cat(pixel_wise_features, 0)
            rois_centers_list = []
            for bs_idx in range(batch_size):
                cur_roi = guided_anchors[bs_idx][:, 0:7].contiguous()
                cur_roi = F.pad(cur_roi, [1, 0],mode='constant',value=bs_idx)
                rois_centers_list.append(cur_roi[:,:4])
            rois_centers = torch.cat(rois_centers_list, dim=0) 
            voxel_wise_features = self.voxel_features_sampling(rois_centers,  points_mean, pointwise)

            l_features = self.fc(l_features)
            point_wise_features = nearest_neighbor_interpolate(rois_centers, l_xyz, l_features)
            ret['node_features'] = torch.cat((pixel_wise_features, voxel_wise_features, point_wise_features),-1)
            ret['node_pos'] = rois_centers
            rpn_outs, state = self.rgnn(ret)
            guided_anchors, bbox_score = self.rpn_head.rcnn_get_guided_anchors(*rpn_outs, guided_anchors, ret['anchors_mask'],
                                                                        None,thr=.1)

        det_bboxes, det_scores = self.extra_head.get_rescore_bboxes(
            guided_anchors, bbox_score, img_meta, self.test_cfg.extra)
        results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]

        return results

from ..utils import change_default_args, Sequential
class Aggregation(nn.Module):
    def __init__(self, num_channel1, num_channel2, out_channel=64, name=""):
        super(Aggregation, self).__init__()
        BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
        self.proj1 = Sequential(
            nn.Linear(num_channel1, out_channel, bias=False),
            BatchNorm1d(out_channel),
            nn.ReLU(),
        )
        self.proj2 = Sequential(
            nn.Linear(num_channel2, out_channel, bias=False),
            BatchNorm1d(out_channel),
            nn.ReLU(),
        )
        self.w1 = nn.Linear(num_channel1, 1, bias=False)
        self.w2 = nn.Linear(num_channel2, 1, bias=False)
        
    def forward(self, align_feature, feature):
        proj1 = self.proj1(align_feature)
        proj2 = self.proj2(feature)
        w1 = self.w1(align_feature)
        w2 = self.w2(feature)
        weights = torch.cat([w1, w2], dim=1)
        weights = torch.softmax(weights, dim=1)
        weights_slice = torch.split(weights, 1, dim=1)
        aggregation = weights_slice[0] * proj1 + weights_slice[1] * proj2
        return aggregation
