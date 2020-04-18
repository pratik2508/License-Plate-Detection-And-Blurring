import argparse
import shutil
import time
from pathlib import Path
from sys import platform
import matplotlib.pyplot as plt
from models import *
from utils.datasets import *
from utils.utils import *
import cv2 as cv
import numpy as np




def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.45,
        # save_txt=False,
        save_txt=True,   # saving corrdinates in text file
        save_images=True
):



    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        print(model)
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()
    print(eval)

    # Set Dataloader
    dataloader = LoadImages(images, img_size=img_size)



    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    print(len(classes))
    # colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
    colors = [[0,51,102] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        print('image %g/%g  %s: ' % (i + 1, len(dataloader),  path), end='')
        save_path = str(Path(output) / Path(path).name)


        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold


        if len(pred) > 0:
            # Run NMS on predictions
            try :
                detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

                # Rescale boxes from 416 to true image size
                detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

                # Print results to screen
                unique_classes = detections[:, -1].cpu().unique()
                for c in unique_classes:
                    n = (detections[:, -1].cpu() == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')

                # Draw bounding boxes and labels of detections
                for crop,(x1, y1, x2, y2, conf, cls_conf, cls) in enumerate(detections):
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write('%g %g %g %g %g %g\n' %
                                    (x1, y1, x2, y2, cls, cls_conf * conf))

                    save_path_new = str(Path(output) /(str(crop)+'_' + Path(path).name))
                    print(int(x1), int(x2))

                    # print(str(str(crop)+'_' + Path(path).name))
                    print(save_path_new)
                    if int(x1) < 5:
                        x1 = 0
                        cropped = im0[int(y1):int(y2), int(x1):int(x2 + 5)]
                    elif (int(x2)+5) > im0.shape[1]:
                        x2 = im0.shape[1]
                        cropped = im0[int(y1):int(y2), int(x1 - 5):int(x2)]
                    else:
                        cropped =im0[int(y1):int(y2), int(x1-5):int(x2+5)]
                    #     cv2.imwrite(save_path_new, cropped)
                    # cropped = im0[int(y1):int(y2), int(x1):int(x2)]
                    # cv2.imwrite(save_path_new, cropped)

                    # add blur to image
                # for i in detection:
                    roi_corners = np.array([[(int(x1-5), int(y1)), (int(x1-5),int(y2)), (int(x2+5),int(y2)),
                                             (int(x2+5), int(y1))]], dtype=np.int32)
                    # img= cv.imread('image.jpeg')
                    blurred_image = cv.GaussianBlur(im0, (9, 9), 6)
                    mask = np.zeros(im0.shape, dtype=np.uint8)
                    channel_count = im0.shape[2]
                    ignore_mask_color = (255,) * channel_count
                    cv.fillPoly (mask, roi_corners, ignore_mask_color)
                    # plt.imshow(mask)
                    # plt.show()
                    mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
                    # plt.imshow(mask_inverse)
                    # plt.show()
                    im0= cv.bitwise_and(blurred_image, mask) + cv.bitwise_and(im0, mask_inverse)
                    # plt.imshow(Final_img)
                    # plt.show()
                    # Add bbox to the image
                    # label = '%s %.2f' % (classes[int(cls)], conf)     # removing text with confidence score
                    label = '%.2f' % (conf)
                # plt.imshow(im0)
                # plt.show()

                # final_img=im0
                #     plot_one_box([x1, y1, x2, y2], im0, color=colors[int(cls)])    # without labels of license plate

                    # plot_one_box([x1, y1, x2, y2], im0, label=label,color=colors[int(cls)])


            except:
                print("sth wrong")

        dt = time.time() - t

        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            if len(pred) > 0:
                cv2.imwrite(save_path, im0)
                # cv2.imwrite(save_path, cropped)


    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/weight.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
