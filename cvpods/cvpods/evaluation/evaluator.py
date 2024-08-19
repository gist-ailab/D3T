#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import datetime
import time
from collections import OrderedDict
from contextlib import contextmanager
from loguru import logger

import torch

from cvpods.utils import comm, log_every_n_seconds

from .registry import EVALUATOR


@EVALUATOR.register()
class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if comm.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, _model_name):
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    os.makedirs(f'/SSDe/heeseon/src/D3T/outputs/visualize/{_model_name}', exist_ok=True)

    def visualize_pred(image_np, instances, idx):
        image_pred = image_np.copy()
        class_names = {0: "person", 1: "car", 2: "bicycle"}    # 'person', 'car', 'bicycle'

        # 바운딩 박스, 클래스, 스코어 정보 추출
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()

        # 바운딩 박스 그리기
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = classes[i]
            class_name = class_names.get(class_id, "Unknown")
            score = scores[i]

            if score < 0.5 : continue
            
            # 바운딩 박스 그리기
            original_h, original_w = 512, 640
            resize_h, resize_w = 800, 1000

            scale_x = resize_w / original_w
            scale_y = resize_h / original_h

            resized_bbox = box.copy()  # 원본을 변경하지 않기 위해 clone 사용
            # bbox의 좌표들을 각각 스케일링
            resized_bbox[0] = box[0] * scale_x  # x1
            resized_bbox[1] = box[1] * scale_y  # y1
            resized_bbox[2] = box[2] * scale_x  # x2
            resized_bbox[3] = box[3] * scale_y  # y2

            x1, y1, x2, y2 = map(int, resized_bbox)
            cv2.rectangle(image_pred, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 클래스와 스코어 텍스트 추가
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image_pred, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # # 이미지 시각화 및 저장
        # plt.imshow(image_pred)
        # plt.axis('off')
        # plt.title("Predicted Instances")
        # plt.savefig(f'/SSDe/heeseon/src/D3T/outputs/visualize/{idx}_prediction.png', bbox_inches='tight', pad_inches=0)
        # # plt.savefig("predicted_instances.png", bbox_inches='tight', pad_inches=0)
        # plt.close()

        return image_pred

    def visualize_gt(image_np, inputs, idx):
        image_gt = image_np.copy()
        class_names = {0: "person", 1: "car", 2: "bicycle"}    # 'person', 'car', 'bicycle'

        gt_boxes = inputs[0]['instances'].gt_boxes.tensor
        gt_classes = inputs[0]['instances'].gt_classes

        # 이미지에 bbox와 클래스 레이블 그리기
        for i in range(len(gt_boxes)):
            box = gt_boxes[i].cpu().numpy().astype(int)
            class_id = gt_classes[i].item()
            class_name = class_names.get(class_id, "Unknown")

            # bbox 그리기
            x1, y1, x2, y2 = box
            cv2.rectangle(image_gt, (x1, y1), (x2, y2), (255, 0, 0), 2) # BGR

            # 클래스 이름 쓰기
            cv2.putText(image_gt, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # # 이미지를 시각화
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(image_gt, cv2.COLOR_BGR2RGB))
        # plt.title("Image with Ground Truth Bounding Boxes")
        # plt.axis("off")
        # # plt.savefig(f'/SSDe/heeseon/src/D3T/outputs/visualize/{idx}_ground_truth.png', bbox_inches='tight', pad_inches=0)
        # plt.close()

        return image_gt

    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger.info("Start inference on {} data samples".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)

    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            ############################    visualize results    ############################
            image_tensor = inputs[0]['image']
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

            # ##  visualize input image  ##
            # plt.imshow(image_np)
            # plt.title('Input Image')
            # plt.axis('off')  # 축 제거
            # plt.savefig(f'/SSDe/heeseon/src/D3T/outputs/visualize/input_image_{idx}.png', bbox_inches='tight', pad_inches=0)
            # plt.close()
            # #############################

            instances = outputs[0]['instances']

            image_pred = visualize_pred(image_np, instances, idx)
            image_gt = visualize_gt(image_np, inputs, idx)

            ######   visualize and save   ######
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))

            # Ground Truth 이미지 표시
            ax[0].imshow(cv2.cvtColor(image_gt, cv2.COLOR_BGR2RGB))
            ax[0].set_title("Ground Truth")
            ax[0].axis("off")

            # Predicted 이미지 표시
            ax[1].imshow(cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB))
            ax[1].set_title("Predicted")
            ax[1].axis("off")

            plt.savefig(f'/SSDe/heeseon/src/D3T/outputs/visualize/{_model_name}/{idx}.png', bbox_inches='tight', pad_inches=0)
            ####################################

            #################################################################################

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    "INFO",
                    "Inference done {}/{}. {:.4f} s / sample. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / sample per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / sample per device, "
        "on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def inference_on_files(evaluator):
    """
    Evaluate the metrics with evaluator on the predicted files

    Args:
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    # num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger.info("Start evaluate on dumped prediction")

    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    start_time = time.perf_counter()
    results = evaluator.evaluate_files()
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info("Total inference time: {}".format(total_time_str))

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
