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

__all__ = [
    'ArcFaceONNX',
]


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
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
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        if self.session is None:
            # ── Tuned SessionOptions for maximum throughput ──
            opts = onnxruntime.SessionOptions()
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern = True
            opts.enable_mem_reuse = True
            opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
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
        self.output_shape = outputs[0].shape

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def get_batch(self, img, faces):
        if len(faces) == 0:
            return []

        aimgs = []
        for face in faces:
            aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
            aimgs.append(aimg)

        # get_feat already supports list of images and batching out of the box
        embeddings = self.get_feat(aimgs)

        for i, face in enumerate(faces):
            face.embedding = embeddings[i].flatten()

        return faces

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        providers = self.session.get_providers()
        use_io_binding = any(p in providers for p in ('CUDAExecutionProvider', 'TensorrtExecutionProvider'))

        if use_io_binding:
            import numpy as np
            io_binding = self.session.io_binding()
            blob_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(blob, 'cuda', 0)
            io_binding.bind_ortvalue_input(self.input_name, blob_ortvalue)
            for out_name in self.output_names:
                io_binding.bind_output(out_name, 'cuda')
            self.session.run_with_iobinding(io_binding)
            net_out = io_binding.get_outputs()[0].numpy()
        else:
            net_out = self.session.run(self.output_names, {self.input_name: blob})[0]

        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out


