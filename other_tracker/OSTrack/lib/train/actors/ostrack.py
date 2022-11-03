from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F

class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data, data_aug=None, data_aug_1=None):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        if data_aug:
            out_dict, x_aug, x_aug_1= self.forward_pass(data, data_aug, data_aug_1)
            x = out_dict['backbone_feat']
        else:
            out_dict = self.forward_pass(data)
        # x_aug = out_dict_aug['backbone_feat']

        # compute losses
        loss, status = self.compute_losses(out_dict, data)
         

        if data_aug:
            loss_mix = self.trackmix_loss(x,x_aug,x_aug_1)
            loss = loss + loss_mix
            status['Loss/track_mix_loss'] =loss_mix.item()
            
        return loss, status

    def forward_pass(self, data, data_aug=None, data_aug_1=None):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1


        template_list = []
        template_list_aug = []
        template_list_aug_1 = []

        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # trackmix 
        if data_aug:
            for i in range(self.settings.num_template):
                template_img_i = data_aug['template_images'][i].view(-1,
                                                                *data_aug['template_images'].shape[2:])  # (batch, 3, 128, 128)
                # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
                template_list_aug.append(template_img_i)

            search_img_aug = data_aug['search_images'][0].view(-1, *data_aug['search_images'].shape[2:])

            for i in range(self.settings.num_template):
                template_img_i = data_aug_1['template_images'][i].view(-1,
                                                                *data_aug_1['template_images'].shape[2:])  # (batch, 3, 128, 128)
                # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
                template_list_aug_1.append(template_img_i)

            search_img_aug_1 = data_aug_1['search_images'][0].view(-1, *data_aug_1['search_images'].shape[2:])


        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        if data_aug and self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z_aug = generate_mask_cond(self.cfg, template_list_aug[0].shape[0], template_list_aug[0].device,
                                            data_aug['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate_aug = adjust_keep_rate(data_aug['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

            box_mask_z_aug_1 = generate_mask_cond(self.cfg, template_list_aug_1[0].shape[0], template_list_aug_1[0].device,
                                            data_aug_1['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate_aug_1 = adjust_keep_rate(data_aug_1['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])



        # trackmix               
        if len(template_list) == 1 and data_aug:
            template_list = template_list[0]
            template_list_aug = template_list_aug[0]
            template_list_aug_1 = template_list_aug_1[0]
        else:
            template_list = template_list[0]


        if data_aug:
            out_dict, x_aug, x_aug_1 = self.net(template=template_list,
                                search=search_img,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                template_aug=template_list_aug,
                                search_aug=search_img_aug,
                                ce_template_mask_aug=box_mask_z_aug,
                                ce_keep_rate_aug=ce_keep_rate_aug,
                                return_last_attn_aug=False,
                                template_aug_1=template_list_aug_1,
                                search_aug_1=search_img_aug_1,
                                ce_template_mask_aug_1=box_mask_z_aug_1,
                                ce_keep_rate_aug_1=ce_keep_rate_aug_1,
                                return_last_attn_aug_1=False,
                                )
       

            return out_dict, x_aug, x_aug_1
        else:
            out_dict = self.net(template=template_list,
                    search=search_img,
                    ce_template_mask=box_mask_z,
                    ce_keep_rate=ce_keep_rate,
                    return_last_attn=False)
            return out_dict

    def compute_losses(self, pred_dict, gt_dict,return_status=True):

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def trackmix_loss(self, x, x_aug, x_aug_1):
        p_feat_clean, p_feat_aug1, p_feat_aug2 = F.softmax(x,dim=1), F.softmax(x_aug,dim=1), F.softmax(x_aug_1,dim=1)
        p_feat_mixture = torch.clamp((p_feat_clean + p_feat_aug1 + p_feat_aug2 ) / 3.,
                            1e-7, 1).log()
        self_id_loss = 12 * (
                            F.kl_div(p_feat_mixture,
                                     p_feat_clean,
                                     reduction='batchmean') +
                            F.kl_div(p_feat_mixture,
                                     p_feat_aug1,
                                     reduction='batchmean') + 
                            F.kl_div(p_feat_mixture,
                                     p_feat_aug2,
                                     reduction='batchmean')
                           ) / 3.
        return self_id_loss
