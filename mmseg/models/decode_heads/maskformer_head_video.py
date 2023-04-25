# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import MaskFormerHead as MMDET_MaskFormerHead
except ModuleNotFoundError:
    MMDET_MaskFormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class MaskFormerHeadVideo(MMDET_MaskFormerHead):
    """Implements the MaskFormer head.

    See `Per-Pixel Classification is Not All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 num_classes: int = 150,
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.out_channels = kwargs['out_channels']
        self.align_corners = True
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_MaskFormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []
        for data_sample in batch_data_samples:
            # Add `batch_input_shape` in metainfo of data_sample, which would
            # be used in MaskFormerHead of MMDetection.
            metainfo = data_sample.metainfo
            metainfo['batch_input_shape'] = metainfo['img_shape']
            data_sample.set_metainfo(metainfo)
            seqlen = data_sample.gt_sem_seg.data.shape[0]
            for i in range(seqlen):
                batch_img_metas.append(data_sample.metainfo)
                gt_sem_seg = data_sample.gt_sem_seg.data[i, ...]
                classes = torch.unique(
                    gt_sem_seg,
                    sorted=False,
                    return_inverse=False,
                    return_counts=False)

                # remove ignored region
                gt_labels = classes[classes != self.ignore_index]

                masks = []
                for class_id in gt_labels:
                    masks.append(gt_sem_seg == class_id)

                if len(masks) == 0:
                    gt_masks = torch.zeros((0, gt_sem_seg.shape[-2],
                                            gt_sem_seg.shape[-1])).to(gt_sem_seg)
                else:
                    gt_masks = torch.stack(masks).squeeze(1)

                instance_data = InstanceData(
                    labels=gt_labels, masks=gt_masks.long())
                batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        
        # * x: (N, L, C, H, W)
        # * batch_data_samples: (N) -> (N*L)
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        # * cls_scores: (num_decoder, N*L, num_queries, cls_out_channels)
        # * all_mask_preds: (num_decoder, N*L, num_queries, H, W)
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        # * batch_img_metas: (N)
        batch_data_samples = []
        for metainfo in batch_img_metas:
            metainfo['batch_input_shape'] = metainfo['img_shape']
            batch_data_samples.append(SegDataSample(metainfo=metainfo))
        # Forward function of MaskFormerHead from MMDetection needs
        # 'batch_data_samples' as inputs, which is image shape　actually.
        # * x: (N, L, C, H, W)
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        # * cls_scores: (N*L, num_queries, cls_out_channels)
        # * all_mask_preds: (N*L, num_queries, H, W)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=img_shape,
            mode='bilinear',
            align_corners=False)

        # semantic inference
        # * cls_scores: (N*L, num_queries, num_cls)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        # * mask_pred: (N*L, num_queries, H, W)
        mask_pred = mask_pred_results.sigmoid()
        # * seg_logits: (N*L, num_cls, H, W)
        seg_logits = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        return seg_logits
    
    def forward(self, x: Tuple[Tensor],
                batch_data_samples: SampleList) -> Tuple[Tensor]:
        """Forward function.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: a tuple contains two elements.

                - all_cls_scores (Tensor): Classification scores for each\
                    scale level. Each is a 4D-tensor with shape\
                    (num_decoder, batch_size, num_queries, cls_out_channels).\
                    Note `cls_out_channels` should includes background.
                - all_mask_preds (Tensor): Mask scores for each decoder\
                    layer. Each with shape (num_decoder, batch_size,\
                    num_queries, h, w).
        """
        # * x: (N, L, C, H, W)
        bs, seqlen = x.shape[0], x.shape[1]
        # * batch_img_metas: (N)
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas) * seqlen
        input_img_h, input_img_w = batch_img_metas[0]['batch_input_shape']
        # * padding_mask: (N*L, H, W)
        padding_mask = x[-1].new_ones((batch_size, input_img_h, input_img_w),
                                      dtype=torch.float32)
        for i in range(batch_size):
            img_h, img_w = batch_img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1), size=x[-1].shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)
        # when backbone is swin, memory is output of last stage of swin.
        # when backbone is r50, memory is output of tranformer encoder.
        # * x: (N, L, C, H, W) -> (N*L, C, H, W)
        x = x.view(-1, *x.shape[2:])
        # * mask_features: (N*L, C, H, W)
        # * memory: (N*L, C, H, W)
        mask_features, memory = self.pixel_decoder(x, batch_img_metas)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        # * memory (N*L, C, H, W) -> (N*L, H*W, C)
        memory = memory.flatten(2).permute(0, 2, 1)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        # * padding_mask (N*L, H*W)
        padding_mask = padding_mask.flatten(1)
        # query_embed = (num_queries, embed_dims)
        query_embed = self.query_embed.weight
        # * query_embed = (N*L, num_queries, embed_dims)
        query_embed = query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        target = torch.zeros_like(query_embed)
        
        # * Reshape all stuffs to (N, L, ...)
        memory = memory.view(bs, seqlen, *memory.shape[1:])
        query_embed = query_embed.view(bs, seqlen, *query_embed.shape[1:])
        pos_embed = pos_embed.view(bs, seqlen, *pos_embed.shape[1:])
        padding_mask = padding_mask.view(bs, seqlen, *padding_mask.shape[1:])
        # * target (N, num_queries, embed_dims)
        target = target.view(bs, seqlen, *target.shape[1:])[:, 0, ...]
        
        for i in range(seqlen):
            # * out_dec: (num_decoder, N*L, num_queries, embed_dims)
            out_dec = self.transformer_decoder(
                query=target,
                key=memory[:, i, ...],
                value=memory[:, i, ...],
                query_pos=query_embed[:, i, ...],
                key_pos=pos_embed[:, i, ...],
                key_padding_mask=padding_mask[:, i, ...])
            target = out_dec[-1]

        # * cls_scores: (num_decoder, N*L, num_queries, cls_out_channels)
        all_cls_scores = self.cls_embed(out_dec)

        # * mask_preds: (num_decoder, N*L, num_queries, C)
        mask_embed = self.mask_embed(out_dec)
        # * all_mask_preds: (num_decoder, N*L, num_queries, H, W)
        all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed,
                                      mask_features)

        return all_cls_scores, all_mask_preds
