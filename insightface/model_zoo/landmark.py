# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import division
import numpy as np
import cv2
import onnx
import onnxruntime
from ..utils import face_align
from ..utils import transform
from ..data import get_object

__all__ = [
    'Landmark',
]


class Landmark:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            #print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid<3 and node.name=='bn_data':
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            opts = onnxruntime.SessionOptions()
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern = True
            opts.enable_mem_reuse = True
            self.session = onnxruntime.InferenceSession(self.model_file, opts)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        self.require_pose = False
        if output_shape[1]==3309:
            self.lmk_dim = 3
            self.lmk_num = 68
            self.mean_lmk = get_object('meanshape_68.pkl')
            self.require_pose = True
        else:
            self.lmk_dim = 2
            self.lmk_num = output_shape[1]//self.lmk_dim
        self.taskname = 'landmark_%dd_%d'%(self.lmk_dim, self.lmk_num)

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def _decode_pred(self, pred, M):
        """Shared decode logic: reshape → scale → invert-transform."""
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)
        IM = cv2.invertAffineTransform(M)
        return face_align.trans_points(pred, IM)

    def get(self, img, face):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        _scale = self.input_size[0] / (max(w, h) * 1.5)
        aimg, M = face_align.transform(img, center, self.input_size[0], _scale, 0)
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]
        pred = self._decode_pred(pred, M)
        face[self.taskname] = pred
        if self.require_pose:
            P = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            s, R, t = transform.P2sRt(P)
            rx, ry, rz = transform.matrix2angle(R)
            face['pose'] = np.array([rx, ry, rz], dtype=np.float32)
        return pred

    def get_batch(self, batch_items):
        """
        Process N faces in ONE ONNX session.run() call.
        batch_items: list of (img, face_obj) tuples.
        Populates face_obj[self.taskname] for each face in-place.
        """
        if not batch_items:
            return

        aimgs, Ms = [], []
        for img, face in batch_items:
            bbox = face.bbox
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            _scale = self.input_size[0] / (max(w, h) * 1.5)
            aimg, M = face_align.transform(img, center, self.input_size[0], _scale, 0)
            aimgs.append(aimg)
            Ms.append(M)

        # Single GPU call for the entire batch → [N, 3, 192, 192]
        blob = cv2.dnn.blobFromImages(aimgs, 1.0/self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        
        providers = self.session.get_providers()
        use_io_binding = any(p in providers for p in ('CUDAExecutionProvider', 'TensorrtExecutionProvider'))

        if use_io_binding:
            io_binding = self.session.io_binding()
            io_binding.bind_cpu_input(self.input_name, blob)
            for name in self.output_names:
                io_binding.bind_output(name)
            self.session.run_with_iobinding(io_binding)
            all_preds = io_binding.copy_outputs_to_cpu()[0]
        else:
            all_preds = self.session.run(self.output_names, {self.input_name: blob})[0]

        for i, (img, face) in enumerate(batch_items):
            pred = self._decode_pred(all_preds[i].copy(), Ms[i])
            face[self.taskname] = pred
            if self.require_pose:
                P = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
                s, R, t = transform.P2sRt(P)
                rx, ry, rz = transform.matrix2angle(R)
                face['pose'] = np.array([rx, ry, rz], dtype=np.float32)
