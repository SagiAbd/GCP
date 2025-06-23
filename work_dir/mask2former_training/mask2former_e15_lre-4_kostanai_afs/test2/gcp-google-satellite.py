auto_scale_lr = dict(base_batch_size=2, enable=False)
backend_args = None
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=True,
        seg_pad_value=255,
        size=(
            512,
            512,
        ),
        type='BatchFixedSizePad'),
]
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=True,
            seg_pad_value=255,
            size=(
                512,
                512,
            ),
            type='BatchFixedSizePad'),
    ],
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_seg=True,
    pad_size_divisor=32,
    seg_pad_value=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
data_root = 'data/google-satellite-test-christchurch/'
dataset_type = 'WHUMixVectorDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True, test_out_dir='visualizations', type='DetVisualizationHook'))
default_scope = 'mmdet'
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
launcher = 'none'
load_from = 'checkpoints/gcp_r50_pretrained_12e_whu-mix-vector.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 8
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_seg=True,
        pad_size_divisor=32,
        seg_pad_value=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    frozen_parameters=[
        'backbone',
        'panoptic_head.pixel_decoder',
        'panoptic_head.transformer_decoder',
        'panoptic_head.decoder_input_projs',
        'panoptic_head.query_embed',
        'panoptic_head.query_feat',
        'panoptic_head.level_embed',
        'panoptic_head.cls_embed',
        'panoptic_head.mask_embed',
    ],
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes=1,
        type='PolyFormerFusionHeadV2'),
    panoptic_head=dict(
        dp_polygonize_head=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=3,
            return_intermediate=True),
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_poly_ang=dict(
            loss_weight=1.0, reduction='mean', type='SmoothL1Loss'),
        loss_poly_reg=dict(
            loss_weight=1.0, reduction='mean', type='SmoothL1Loss'),
        num_queries=300,
        num_stuff_classes=0,
        num_things_classes=1,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='MSDeformAttnPixelDecoder'),
        poly_cfg=dict(
            align_iou_thre=0.5,
            apply_angle_loss=True,
            apply_prim_pred=True,
            lam=4,
            loss_weight_dp=0.01,
            map_features=True,
            mask_cls_thre=0.0,
            max_align_dis=15,
            max_match_dis=10,
            max_offsets=5,
            max_step_size=128,
            num_cls_channels=2,
            num_inter_points=64,
            num_min_bins=32,
            point_as_prim=True,
            poly_decode_type='dp',
            polygonize_mode='cv2_single_mask',
            polygonized_scale=4.0,
            pred_angle=False,
            prim_cls_thre=0.1,
            proj_gt=False,
            reg_targets_type='vertice',
            return_poly_json=False,
            sample_points=True,
            step_size=4,
            stride_size=64,
            use_coords_in_poly_feat=True,
            use_decoded_feat_in_poly_feat=True,
            use_gt_jsons=False,
            use_ind_offset=True,
            use_point_feat_in_poly_feat=True,
            use_ref_rings=False),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True),
        type='PolygonizerHead'),
    test_cfg=dict(
        filter_low_score=False,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=300,
        panoptic_on=False,
        semantic_on=False),
    train_cfg=dict(
        add_target_to_data_samples=True,
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        prim_assigner=dict(
            match_costs=[
                dict(type='PointL1Cost', weight=0.1),
            ],
            type='HungarianAssigner'),
        sampler=dict(type='MaskPseudoSampler')),
    type='PolyFormerV2')
num_classes = 1
num_stuff_classes = 0
num_things_classes = 1
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=1e-05,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=0.01),
            level_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_feat=dict(decay_mult=0.0, lr_mult=1.0)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=8,
        gamma=0.1,
        milestones=[
            6,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root='data/google-satellite-test-christchurch/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=False,
                with_mask=True,
                with_poly_json=False),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='WHUMixVectorDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file='data/google-satellite-test/test.json',
        backend_args=None,
        calculate_iou_ciou=True,
        calculate_mta=True,
        mask_type='polygon',
        metric=[
            'segm',
        ],
        score_thre=0.5,
        type='CocoMetric'),
]
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=False,
        with_mask=True,
        with_poly_json=False),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=8, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='train/train.json',
        backend_args=None,
        data_prefix=dict(img='train/images'),
        data_root='data/google-satellite-test-christchurch/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                with_poly_json=False),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='RandomFlip'),
            dict(prob=0.75, type='Rotate90'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        type='WHUMixVectorDataset'),
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        with_poly_json=False),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='RandomFlip'),
    dict(prob=0.75, type='Rotate90'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val/val.json',
        backend_args=None,
        data_prefix=dict(img='val/images'),
        data_root='data/google-satellite-test-christchurch/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=False,
                with_mask=True,
                with_poly_json=False),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='WHUMixVectorDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file='data/google-satellite-test/test.json',
        backend_args=None,
        calculate_iou_ciou=True,
        calculate_mta=True,
        mask_type='polygon',
        metric=[
            'segm',
        ],
        score_thre=0.5,
        type='CocoMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TanmlhVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dir\\mask2former_training\\mask2former_e15_lre-4_kostanai_afs\\test2'
