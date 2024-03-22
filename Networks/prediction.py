import torch
import numpy as np
from numpy import pad
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw, ImageFont
from Networks.yolov4.YOLOV4 import non_max_suppression
import colorsys
import cv2
import sys



def cvTopil(cv):
    cv = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(cv)
    return pil
def pilTocv(pil):
    cv = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return cv

'''
模型接口类
传入numpy类型
传返回cv类型
'''
class Predict:
    def __init__(self):
        self.status = 'Classification'
        print('加载模型')
        self.classification  = torch.load("./CSSSAN.pth", map_location='cpu')
        self.ObjectDetection = torch.load("./YOLO.pth",map_location="cpu")
        self.ChangeDetection = torch.load('./STANet.pth', map_location='cpu')
        print('加载完成')

        self.finish_flag = False

    def classification_pre(self,img1,img2=None):
        Data = img1
        Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
        Data = Data.astype(np.float32)

        # 归一化
        for band in range(Band):
            Data[:, :, band] = (Data[:, :, band] - np.min(Data[:, :, band])) / (
                    np.max(Data[:, :, band]) - np.min(Data[:, :, band]))

        #填充
        patch_size = 9
        Data_Padding = np.zeros((Height + int(patch_size - 1), Width + int(patch_size - 1), Band))
        for band in range(Band):
            Data_Padding[:, :, band] = pad(Data[:, :, band], int((patch_size - 1) / 2), 'symmetric')

        outputs = np.zeros((Height, Width))
        for i in range(Height):
            for j in range(Width):
                print(i)
                image_patch = Data_Padding[i:i + patch_size, j:j + patch_size, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2))
                prediction = self.classification(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction

        img = np.zeros((Height, Width, 3), dtype=np.uint8) #结果是cv类型
        colorlist = [[00, 00, 00], [0xFF, 0x66, 00], [0xFF, 0xFF, 00], [0x99, 0xFF, 00], [0x00, 0xFF, 0xFF],
                     [0x00, 0x66, 0xFF], [0xFF, 00, 0xFF], [0xFF, 00, 0x33], [0x2B, 0x6F, 0xD5], [0xBD, 0x1A, 0xE6],
                     [0xE6, 0, 0], [0x6B, 0xE6, 0x1A], [0xDD, 0xB8, 0x22], [0x1A, 0xE, 0xE6], [0x55, 0x55, 0xAA],
                     [0x55, 0xAA, 0x99]]

        for i in range(Height):
            for j in range(Width):
                img[i][j][0] = colorlist[int(outputs[i][j])][0]
                img[i][j][1] = colorlist[int(outputs[i][j])][1]
                img[i][j][2] = colorlist[int(outputs[i][j])][2]

        self.finish_flag = True
        return img

    def objectdetection_pre(self,img1,img2=None):
        input_shape = [412, 412]
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        anchors = np.array(
            [(10, 10), (25, 13), (15, 24), (31, 20), (25, 29), (29, 62), (47, 39), (65, 69), (118, 138)])  # 锚框中心
        bbox_attrs = 20  # bbox_attrs     = 5 + num_classes
        num_classes = 15

        letterbox_image = False  # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        count = False  # count               指定了是否进行目标的计数
        confidence = 0.5  # 只有得分大于置信度的预测框会被保留下来
        nms_iou = 0.0  # 非极大抑制所用到的nms_iou大小
        class_names = ["plane",
                       "ship",
                       "storage-tank",
                       "baseball-diamond",
                       "tennis-court",
                       "basketball-court",
                       "ground-track-field",
                       "harbor",
                       "bridge",
                       "large-vehicle",
                       "small-vehicle",
                       "helicopter",
                       "roundabout",
                       "soccer-ball-field",
                       "swimming-pool"]

        img1 = cvTopil(img1)
        image = img1.resize((412, 412))
        print('here')
        image_to_tensor = transforms.ToTensor()  # 图片转向量
        T_image = image_to_tensor(image)
        out_image = self.ObjectDetection(torch.unsqueeze(T_image, dim=0))  # 输出结果

        outputs = []
        for i, input in enumerate(out_image):
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            # -----------------------------------------------#
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)
            # -----------------------------------------------#
            #   输入为416x416时
            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = input_shape[0] / input_height
            stride_w = input_shape[1] / input_width
            # -------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            # -------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              anchors[anchors_mask[i]]]
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #   batch_size, 3, 52, 52, 85
            # -----------------------------------------------#
            prediction = input.view(batch_size, len(anchors_mask[i]),
                                    bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
            # -----------------------------------------------#
            #   先验框的中心位置的调整参数
            # -----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            # -----------------------------------------------#
            #   先验框的宽高调整参数
            # -----------------------------------------------#
            w = prediction[..., 2]
            h = prediction[..., 3]
            # -----------------------------------------------#
            #   获得置信度，是否有物体
            # -----------------------------------------------#
            conf = torch.sigmoid(prediction[..., 4])
            # -----------------------------------------------#
            #   种类置信度
            # -----------------------------------------------#
            pred_cls = torch.sigmoid(prediction[..., 5:])
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,13,13
            # ----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)
            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,13,13
            # ----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            # ----------------------------------------------------------#
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            # ----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            # ----------------------------------------------------------#
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_classes)), -1)
            outputs.append(output.data)
        # print(outputs)
        results = non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape,
                                      input_shape, letterbox_image, conf_thres=confidence, nms_thres=nms_iou)

        if results[0] is None:
            self.finish_flag = True
            return pilTocv(image)

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]
        font = ImageFont.truetype(font="model_data/simhei.ttf", size=np.floor(3e-2 * image.size[1] + 0.5).astype(
            'int32'))  # """'model_data/simhei.ttf'"""
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        # print(top_label)
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        self.finish_flag = True
        return pilTocv(image)

    def changedetection_pre(self,img1,img2=None):
        print('进入变化检测函数')
        image_to_tensor = transforms.ToTensor()  # 图片转向量
        image1 = image_to_tensor(img1)
        image1 = image1.view(1, 3, 256, 256)

        image2 = image_to_tensor(img2)
        image2 = image2.view(1, 3, 256, 256)

        result = self.ChangeDetection(image1, image2)
        result = torch.argmax(result, dim=1)
        result = result.view(256, 256, 1)
        result = result.to(torch.device('cpu')).detach().numpy()
        result = result * 255

        result = np.concatenate([result, result, result], axis=2)
        result = np.uint8(result)

        self.finish_flag = True
        return  result #返回cv类型


if __name__ == '__main__':
    import scipy
    predict = Predict()
    #测试1
    imageDir = "./img1.mat"
    input = scipy.io.loadmat(imageDir)['salinas_corrected']
    result = predict.classification_pre(input)
    print(type(result))
    cv2.imshow('result', result)
    cv2.waitKey()

    # #测试2
    # input = cv2.imread("./0025.jpg")
    # print(input.shape)
    # result = predict.objectdetection_pre(cvTopil(input))
    # print(type(result))
    # cv2.imshow('result',result)
    # cv2.waitKey()
    #测试3

