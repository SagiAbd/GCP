# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence
import pdb

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.datasets.samplers import TrackImgSampler
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from mmdet.visualization.palette import _get_adaptive_scales


@HOOKS.register_module()
class DetVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            if osp.isabs(self.test_out_dir):
                mkdir_or_exist(self.test_out_dir)
            else:
                self.test_out_dir = osp.join(runner.work_dir, self.test_out_dir)
                mkdir_or_exist(self.test_out_dir)

        # Limit the number of images saved to show_dir
        max_vis_images = 15
        for i, data_sample in enumerate(outputs):
            if i >= max_vis_images:
                break
            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)


@HOOKS.register_module()
class TrackVisualizationHook(Hook):
    """Tracking Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        frame_interval (int): The interval of visualization. Defaults to 30.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict): Arguments to instantiate a file client.
            Defaults to ``None``.
    """

    def __init__(self,
                 draw: bool = False,
                 frame_interval: int = 30,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None) -> None:
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.frame_interval = frame_interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self.image_idx = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[TrackDataSample]) -> None:
        """Run after every ``self.interval`` validation iteration.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        assert len(outputs) == 1, \
            'only batch_size=1 is supported while validating.'

        sampler = runner.val_dataloader.sampler
        if isinstance(sampler, TrackImgSampler):
            if self.every_n_inner_iters(batch_idx, self.frame_interval):
                total_curr_iter = runner.iter + batch_idx
                track_data_sample = outputs[0]
                self.visualize_single_image(track_data_sample[0],
                                            total_curr_iter)
        else:
            # video visualization DefaultSampler
            if self.every_n_inner_iters(batch_idx, 1):
                track_data_sample = outputs[0]
                video_length = len(track_data_sample)

                for frame_id in range(video_length):
                    if frame_id % self.frame_interval == 0:
                        total_curr_iter = runner.iter + self.image_idx + \
                                          frame_id
                        img_data_sample = track_data_sample[frame_id]
                        self.visualize_single_image(img_data_sample,
                                                    total_curr_iter)
                self.image_idx = self.image_idx + video_length

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[TrackDataSample]) -> None:
        """Run after every testing iteration.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        assert len(outputs) == 1, \
            'only batch_size=1 is supported while testing.'

        if self.test_out_dir is not None:
            if osp.isabs(self.test_out_dir):
                mkdir_or_exist(self.test_out_dir)
            else:
                self.test_out_dir = osp.join(runner.work_dir, self.test_out_dir)
                mkdir_or_exist(self.test_out_dir)

        sampler = runner.test_dataloader.sampler
        if isinstance(sampler, TrackImgSampler):
            if self.every_n_inner_iters(batch_idx, self.frame_interval):
                track_data_sample = outputs[0]
                self.visualize_single_image(track_data_sample[0], batch_idx)
        else:
            # video visualization DefaultSampler
            if self.every_n_inner_iters(batch_idx, 1):
                track_data_sample = outputs[0]
                video_length = len(track_data_sample)

                for frame_id in range(video_length):
                    if frame_id % self.frame_interval == 0:
                        img_data_sample = track_data_sample[frame_id]
                        self.visualize_single_image(img_data_sample,
                                                    self.image_idx + frame_id)
                self.image_idx = self.image_idx + video_length

    def visualize_single_image(self, img_data_sample: DetDataSample,
                               step: int) -> None:
        """
        Args:
            img_data_sample (DetDataSample): single image output.
            step (int): The index of the current image.
        """
        img_path = img_data_sample.img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        out_file = None
        if self.test_out_dir is not None:
            video_name = img_path.split('/')[-3]
            mkdir_or_exist(osp.join(self.test_out_dir, video_name))
            out_file = osp.join(self.test_out_dir, video_name,
                                osp.basename(img_path))

        self._visualizer.add_datasample(
            osp.basename(img_path) if self.show else 'test_img',
            img,
            data_sample=img_data_sample,
            show=self.show,
            wait_time=self.wait_time,
            pred_score_thr=self.score_thr,
            out_file=out_file,
            step=step)


def draw_all_character(visualizer, characters, w):
    start_index = 2
    y_index = 5
    for char in characters:
        if isinstance(char, str):
            visualizer.draw_texts(
                str(char),
                positions=np.array([start_index, y_index]),
                colors=(0, 0, 0),
                font_families='monospace')
            start_index += len(char) * 8
        else:
            visualizer.draw_texts(
                str(char[0]),
                positions=np.array([start_index, y_index]),
                colors=char[1],
                font_families='monospace')
            start_index += len(char[0]) * 8

        if start_index > w - 10:
            start_index = 2
            y_index += 15

    drawn_text = visualizer.get_image()
    return drawn_text


@HOOKS.register_module()
class GroundingVisualizationHook(DetVisualizationHook):

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            if osp.isabs(self.test_out_dir):
                mkdir_or_exist(self.test_out_dir)
            else:
                self.test_out_dir = osp.join(runner.work_dir, self.test_out_dir)
                mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            data_sample = data_sample.cpu()

            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            text = data_sample.text
            if isinstance(text, str):  # VG
                gt_instances = data_sample.gt_instances
                tokens_positive = data_sample.tokens_positive
                if 'phrase_ids' in data_sample:
                    # flickr30k
                    gt_labels = data_sample.phrase_ids
                else:
                    gt_labels = gt_instances.labels
                gt_bboxes = gt_instances.get('bboxes', None)
                if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
                    gt_instances.bboxes = gt_bboxes.tensor
                print(gt_labels, tokens_positive, gt_bboxes, img_path)
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > self.score_thr]
                pred_labels = pred_instances.labels
                pred_bboxes = pred_instances.bboxes
                pred_scores = pred_instances.scores

                max_label = 0
                if len(gt_labels) > 0:
                    max_label = max(gt_labels)
                if len(pred_labels) > 0:
                    max_label = max(max(pred_labels), max_label)

                max_label = int(max(max_label, 0))
                palette = np.random.randint(0, 256, size=(max_label + 1, 3))
                bbox_palette = [tuple(c) for c in palette]
                # bbox_palette = get_palette('random', max_label + 1)
                if len(gt_labels) >= len(pred_labels):
                    colors = [bbox_palette[label] for label in gt_labels]
                else:
                    colors = [bbox_palette[label] for label in pred_labels]

                self._visualizer.set_image(img)

                for label, bbox, color in zip(gt_labels, gt_bboxes, colors):
                    self._visualizer.draw_bboxes(
                        bbox, edge_colors=color, face_colors=color, alpha=0.3)
                    self._visualizer.draw_bboxes(
                        bbox, edge_colors=color, alpha=1)

                drawn_img = self._visualizer.get_image()

                new_image = np.ones(
                    (100, img.shape[1], 3), dtype=np.uint8) * 255
                self._visualizer.set_image(new_image)

                if tokens_positive == -1:  # REC
                    gt_tokens_positive = [[]]
                else:  # Phrase Grounding
                    gt_tokens_positive = [
                        tokens_positive[label] for label in gt_labels
                    ]
                split_by_character = [char for char in text]
                characters = []
                start_index = 0
                end_index = 0
                for w in split_by_character:
                    end_index += len(w)
                    is_find = False
                    for i, positive in enumerate(gt_tokens_positive):
                        for p in positive:
                            if start_index >= p[0] and end_index <= p[1]:
                                characters.append([w, colors[i]])
                                is_find = True
                                break
                        if is_find:
                            break
                    if not is_find:
                        characters.append([w, (0, 0, 0)])
                    start_index = end_index

                drawn_text = draw_all_character(self._visualizer, characters,
                                                img.shape[1])
                drawn_gt_img = np.concatenate((drawn_img, drawn_text), axis=0)

                self._visualizer.set_image(img)

                for label, bbox, color in zip(pred_labels, pred_bboxes,
                                              colors):
                    self._visualizer.draw_bboxes(
                        bbox, edge_colors=color, face_colors=color, alpha=0.3)
                    self._visualizer.draw_bboxes(
                        bbox, edge_colors=color, alpha=1)
                print(pred_labels, pred_bboxes, pred_scores, colors)
                areas = (pred_bboxes[:, 3] - pred_bboxes[:, 1]) * (
                    pred_bboxes[:, 2] - pred_bboxes[:, 0])
                scales = _get_adaptive_scales(areas)
                score = [str(round(s.item(), 2)) for s in pred_scores]
                font_sizes = [int(13 * scales[i]) for i in range(len(scales))]
                self._visualizer.draw_texts(
                    score,
                    pred_bboxes[:, :2].int(),
                    colors=(255, 255, 255),
                    font_sizes=font_sizes,
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }] * len(pred_bboxes))

                drawn_img = self._visualizer.get_image()

                new_image = np.ones(
                    (100, img.shape[1], 3), dtype=np.uint8) * 255
                self._visualizer.set_image(new_image)
                drawn_text = draw_all_character(self._visualizer, characters,
                                                img.shape[1])
                drawn_pred_img = np.concatenate((drawn_img, drawn_text),
                                                axis=0)
                drawn_img = np.concatenate((drawn_gt_img, drawn_pred_img),
                                           axis=1)

                if self.show:
                    self._visualizer.show(
                        drawn_img,
                        win_name=osp.basename(img_path),
                        wait_time=self.wait_time)
                if out_file is not None:
                    mmcv.imwrite(drawn_img[..., ::-1], out_file)
                else:
                    self.add_image('test_img', drawn_img, self._test_index)
            else:  # OD
                self._visualizer.add_datasample(
                    osp.basename(img_path) if self.show else 'test_img',
                    img,
                    data_sample=data_sample,
                    show=self.show,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    out_file=out_file,
                    step=self._test_index)
