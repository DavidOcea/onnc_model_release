import cv2
import math
import numpy as np
import argparse
from utils import COCO_CLASS

import onnx
import onnxruntime

COCO_CATEGORIES = COCO_CLASS().COCO_CATEGORIES

class Detector():
    def __init__(self, input_shape=(544, 960), prob_threshold=0.4, iou_threshold=0.3, model_path="detection_model.onnx", num_classes=4):
        self.num_classes = num_classes
        self.classes = [clss["name"] for clss in COCO_CATEGORIES]

        self.strides = (4, 16, 32, 64)
        self.input_shape = input_shape
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)
        self.net_session = onnxruntime.InferenceSession(model_path) 

        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid((math.ceil(self.input_shape[0] / self.strides[i]), math.ceil(self.input_shape[1] / self.strides[i])), self.strides[i])
            self.mlvl_anchors.append(anchors)

    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        cx = xv + stride//2
        cy = yv + stride//2
        return np.stack((cx, cy), axis=-1) 

    def _normalize(self, img):
        img = img.astype(np.float32)
        img /= 255
        img = (img - self.mean) / self.std
        return img

    def eqratio_resize(self, srcimg, keep_ratio=True, padding_value=[0,0,0]):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        img_h, img_w, img_c = srcimg.shape
        src_ratio = float(img_w) / img_h
        ex_ratio = float(neww) / newh
        if (ex_ratio < src_ratio):
            resize_ratio = float(neww) / img_w
        else:
            resize_ratio = float(newh) / img_h
        resize_img_w = int(img_w * resize_ratio + 0.5)
        resize_img_h = int(img_h * resize_ratio + 0.5)
        img = cv2.resize(srcimg, (resize_img_w, resize_img_h), interpolation=cv2.INTER_NEAREST)
        pad_w_left = (neww - resize_img_w) // 2
        pad_w_right = neww - resize_img_w - pad_w_left
        pad_h_top = (newh - resize_img_h) // 2
        pad_h_bottom = newh - resize_img_h - pad_h_top
        print("padding value(tblr):",pad_h_top,pad_h_bottom,pad_w_left,pad_w_right)
        #常数填充
        img = cv2.copyMakeBorder(img,pad_h_top,pad_h_bottom,pad_w_left,pad_w_right,cv2.BORDER_CONSTANT,value=padding_value)
        return img, newh, neww, top, left

        
    def detect(self, srcimg):
        img, newh, neww, top, left = self.eqratio_resize(srcimg)
        print(".eqratio_resize shape:", img.shape)
        drawimg = img.copy()
        img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(img)

        # HWC to NCHW 
        input_img = np.transpose(img, [2, 0, 1]) 
        input_img = np.expand_dims(input_img, 0)
        ort_inputs = {'input.1': input_img} 
        outs = self.net_session.run(['1036', '1039','1042', '1045', '1048', '1051', '1054', '1057', '1060', '1063', '1066', '1069'], ort_inputs)
        
        # Runs the forward pass to get output of the output layers
        det_bboxes, det_conf, det_classid = self.post_process(outs)
        
        if det_bboxes.ndim==3:
            det_bboxes = det_bboxes.squeeze(axis=1)
        if det_conf.ndim==2:
            det_conf = det_conf.squeeze(axis=1)
        if det_classid.ndim==2:
            det_classid = det_classid.squeeze(axis=1)

        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int(det_bboxes[i,0]), 0), max(int(det_bboxes[i,1]), 0), min(int(det_bboxes[i,2]), neww), min(int(det_bboxes[i,3]), newh)
            self.drawPred(drawimg, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)
        return drawimg

    def post_process(self, preds, fpn_num=4):
        cls_scores, bbox_preds, quality_preds = preds[:fpn_num], preds[fpn_num:fpn_num*2], preds[fpn_num*2:]
        det_bboxes, det_conf, det_classid = self.get_result(cls_scores, bbox_preds, quality_preds, 1, rescale=False)
        return det_bboxes.astype(np.int32), det_conf, det_classid

    def get_result(self, cls_scores, bbox_preds, quality_preds, scale_factor, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred, qual_pred, anchors in zip(self.strides, cls_scores, bbox_preds, quality_preds, self.mlvl_anchors):
            if cls_score.ndim==3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim==3:
                bbox_pred = bbox_pred.squeeze(axis=0)
            if qual_pred.ndim==3:
                qual_score = qual_pred.squeeze(axis=0)

            bboxes = self.distance2bbox(anchors, bbox_pred*stride, max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score*qual_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)
        if len(indices)>0:
            mlvl_bboxes = mlvl_bboxes[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            return mlvl_bboxes, confidences, classIds
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
        return frame

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='test_imgs/000000000139.jpg', help="image path")
    parser.add_argument('--input_shape', default=(608, 608), type=int, help='input image shape')
    parser.add_argument('--confThreshold', default=0.2, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--model_path', type=str, default='onnx_model/detection_coco_model.onnx', help="model path")
    parser.add_argument('--num_classes', default=80, type=int, help='num_classes')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    net = Detector(input_shape=args.input_shape, 
                    prob_threshold=args.confThreshold, 
                    iou_threshold=args.nmsThreshold,
                    model_path=args.model_path,
                    num_classes=args.num_classes )
    import time
    a = time.time()
    srcimg = net.detect(srcimg)
    b = time.time()
    print('waste time {} s'.format(b-a))
    cv2.imwrite("onnx_result.jpg", srcimg)