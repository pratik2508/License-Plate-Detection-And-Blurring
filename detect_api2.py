import argparse
import shutil
import time
from models import *
from utils.datasets import *
from utils.utils import *
import cv2 as cv
import numpy as np
import traceback
from flask import Flask, request, Response
import psycopg2
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
from logging import Formatter
import detect_batch2
import json
import os

handler = TimedRotatingFileHandler('log-traffic', when='MIDNIGHT', backupCount=7)
formatter = Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
logger = logging.getLogger('werkzeug')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


app = Flask(__name__)
@app.route('/')
def message():
    return("Welcome to License Plate Detection")

@app.route('/LicensePlate',methods=['POST'])  # Single Api
def LP():
    print ("Welcome to License Plate Detection")
    cfg='/mnt/vol1/License-plate-detection/cfg/yolov3.cfg'
    weights=r'/mnt/vol1/License-plate-detection/weights/weight.pt'
    nms_thres = 0.45
    conf_thres = 0.09
    img_size =416

    try:
        imagepath = request.form.get('file')  # Single image path

    except Exception as e:
        logging.error(
            msg='Image File not Found' + str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    try:
        # Initialize model
        device = torch_utils.select_device()
        model = Darknet(cfg, img_size)
        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
            # print(model)
        else:  # darknet format
            load_darknet_weights(model, weights)

        model.to(device).eval()
        # print(eval)

    except Exception as e:
        logger.error(msg='Model Weights Not Found\t' + str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    try:

        # Set Dataloader
        img_load = LoadImages(imagepath, img_size=img_size)
        # Get classes and colors
        classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
        print(len(classes))

    except Exception as e:
        logger.error(msg='error Dataloader OR coco.data Not found\t' + str(e) + '\tTraceback\t' + '~'.join(
            str(traceback.format_exc()).split('\n')))
    final_coordinates = []

    try:
        imgp,img,im0=next(iter(img_load))
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        # print(img.shape)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold
        if len(pred) > 0:

                    detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
                    # Rescale boxes from 416 to true image size
                    detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

                    # Print results to screen
                    unique_classes = detections[:, -1].cpu().unique()
                    for c in unique_classes:
                        n = (detections[:, -1].cpu() == c).sum()
                        print('%g %ss' % (n, classes[int(c)]))

                    # Draw bounding boxes and labels of detections
                    for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                        # if save_txt:  # Write to file
                        #     with open(save_path + '.txt', 'a') as file:
                        #         file.write('%g %g %g %g %g %g\n' %
                        #                    (x1, y1, x2, y2, cls, cls_conf * conf))


                        roi_corners = np.array([[(int(x1 - 5), int(y1)), (int(x1 - 5), int(y2)), (int(x2 + 5), int(y2)),
                                                 (int(x2 + 5), int(y1))]], dtype=np.int32)
                        # coord=to(roi_corners)
                        coordinates=roi_corners.tolist()
                        final_coordinates.append(coordinates[0])
                        # print(final_coordinates)


    except Exception as e:
        logger.error(msg='License Plate Not found' + str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))




    new_final={}
    for i,j in enumerate(final_coordinates):
        new_final[i]=j
    return Response(json.dumps(new_final), status=200)







@app.route('/LicensePlate-Batch/stop', methods=['POST'])
def stop():
    """
    Entry point for stopping processing of a path

    # Arguments :
    :param task_id : Task ID of task to stop
    :return:
    Status 200
    """
    try:
        # task_id = request.args['task_id']

        print("Welcome to License Plate Stop Detection")
        input_path = request.form.get('input_path')
        print('input_path :-' + input_path)

        conn = psycopg2.connect(host="10.1.1.168", database="License_Plate", user="postgres", password="postgres")
        cur = conn.cursor()
        # connection for ai-ml_hub db
        conn1 = psycopg2.connect(host="10.1.1.168", database="ai_ml_hub", user="postgres", password="postgres")
        cur1 = conn1.cursor()
        try:
            cur.execute("Select task_id from catalog.task_details where folder_path='{path}'".format(path=input_path))
            task_id = cur.fetchone()[0]
            print(task_id)

        except Exception as e:

            logger.error(msg='Incorrect input Folder_path\t' + str(e) + '\tTraceback\t' + '~'.join(
                    str(traceback.format_exc()).split('\n')))
            return Response('Incorrect input Folder_path', status=500)


        # Terminate task
        # revoke(task_id, terminate=True)
        detect_batch2.detect_batch1.AsyncResult(task_id).revoke(terminate=True, signal='SIGKILL')

        # queries for table updates on interrupt
        try:
            cur.execute(
                "Select route from catalog.lp_folder_path_info where folder_path='{path}' and state = 1"
                    .format(path=input_path))
            route = cur.fetchone()[0]

            cur.execute("UPDATE catalog.lp_folder_Path_info SET state='2' WHERE route='{route}' and state = 1"
                        .format(route=route))

        except Exception as e:
            logger.error(msg='input path does not have a task in running status \t' + str(e) + '\tTraceback\t' + '~'.join(
                str(traceback.format_exc()).split('\n')))
            return Response('input path does not have a task in running status', status=500)


        ##wrapper use
        cur1.execute(
            "UPDATE catalog.gpu_server_info  SET proc_count = proc_count - 1 WHERE server_name='Server_3'")

        cur.execute("Delete from catalog.task_details where task_id='{task}'".format(task=task_id))

        conn.commit()
        conn.close()
        conn1.commit()
        conn1.close()

        return Response('Batch processing task stopped', status=200)





    except Exception as e:
        logger.error(
            msg='Exception occurred\t' + str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

        return Response('', status=500)


@app.route('/LicensePlate-Blur', methods=['POST'])
def blurring():

    print ("Welcome to Blur License Plate Detection")
    input_path = request.form.get('input_path')
    print('input_path :-' + input_path)

    # input_path = request.form.get('input_path').replace('%20', ' ').replace('\\', '/').replace('//', '/').replace(
    #     '/10.1.1.', '/mnt/10.1.1.')
    # output_path = request.form.get('output_path')

    output_path = request.form.get('output_path')
    print('output_path :-' + output_path)

    range_start = int(request.form.get('range_start'))
    range_end = int(request.form.get('range_end'))
    print(range_start, range_end)

    try:

        task_blur = detect_batch2.detect_blur1.delay(input_path, output_path, range_start, range_end)

        return Response('Blurring_task_id :- '+task_blur.id, status=200)

    except Exception as e:
        logger.error(
            msg='Blurring Not completed'+ str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

        return Response('Blurring Not completed',status=500)


def blur(input_path, output_path, range_start, range_end):   # blur function to be used with api and celery

    try:
        # input_path1 parameter is for ubuntu
        # input_path parameter  is for windows path
        input_path1 = input_path.replace('%20', ' ').replace('\\', '/').replace('//', '/').replace(
            '/10.1.1.', '/mnt/vol1/10.1.1.')
        print(input_path1)

        if os.path.isdir(input_path1):
            print("input path exists")
        else:
            print('input path does not exist')

    except Exception as e:
        print(e)
        logger.error(msg='Input path does not exist or not mounted ' + str(e) + '\tTraceback\t' + '~'.join(
            str(traceback.format_exc()).split('\n')))

    try:
        # output_path1 parameter is for ubuntu
        # output_path parameter  is for windows path
        output_path1 = output_path.replace('%20', ' ').replace('\\', '/').replace('//', '/').replace(
            '/10.1.1.', '/mnt/vol1/10.1.1.')
        print(output_path1)

        if os.path.isdir(output_path1):
            print("output path exists")
        else:
            print('output path does not exist')

    except Exception as e:
        print(e)
        logger.error(msg='output path does not exist or not mounted' + str(e) + '\tTraceback\t' + '~'.join(
            str(traceback.format_exc()).split('\n')))

    try:
        conn = psycopg2.connect(host="10.1.1.168", database="License_Plate", user="postgres", password="postgres")
        cur = conn.cursor()

    except Exception as e:
        logger.error(msg='Connection not established ' + str(e) + '\tTraceback\t' + '~'.join(
            str(traceback.format_exc()).split('\n')))

    try:

        cur.execute("select route from catalog.lp_folder_path_info where folder_path='{path}' order by sno desc"
                    .format(path=input_path))

        var1 = cur.fetchone()
        route = var1[0]
        print(route)

        images_dir = os.listdir(input_path1)
        images_dir.sort()

        image_format = ['.jpg', '.jpeg', '.png', '.tif']
        images_list = []

        for iter in images_dir:
            if iter[-4:] in image_format:
                images_list.append(iter)

    except Exception as e:
        print(e)
        logger.error(msg='Incorrect Image format/n  image should be in .jpg, .jpeg, .png, .tif formats' + str(e)
                         + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    try:
        if (range_end <= len(images_dir)) and (range_end > 0):
            print("valid range_provided")

    except Exception as e:
        logger.error(msg='Invalid range provided' + str(e)
                         + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    try:
        # range is being set on images in images_list if range is  provided for blurring
        if (range_start != None) and (range_end != None):
            images_list = images_list[range_start-1: range_end+1]
            print(len(images_list))

        for k in images_list:
            img = cv2.imread(os.path.join(input_path1, k))

            cur.execute("Select image_name,box from public.{route} where image_name='{i}' and Flag='T'"
                        "order by image_name".format(route=route, i=k))
            result = cur.fetchall()

            cur.execute(
                "Select image_name,box from public.{route}_undetect where image_name='{i}' order by image_name"
                        .format(route=route, i=k))
            res_new = cur.fetchall()
            print(res_new)

            for i in res_new: # for merging
                result.append(i)

            print(result)
            if len(result) > 0:
                for i in result:
                    list = ([int(j) for j in i[1][1:-1].split(',')])
                    roi_corners = np.array([[(list[0], list[1]), (list[0], list[3]), (list[2], list[3]), (list[2], list[1])]],dtype=np.int32)
                    # print(roi_corners)
                    # print(cor_list)
                    # roi_corners = np.array((cor_list), dtype=np.int32)

                    blurred_image = cv.GaussianBlur(img, (9, 9), 11)
                    mask = np.zeros(img.shape, dtype=np.uint8)
                    channel_count = img.shape[2]
                    ignore_mask_color = (255,) * channel_count
                    cv.fillPoly(mask, roi_corners, ignore_mask_color)

                    # plt.imshow(mask)
                    # plt.show()
                    mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
                    img = cv.bitwise_and(blurred_image, mask) + cv.bitwise_and(img, mask_inverse)
                # print(output_path1)
                cv.imwrite(os.path.join(output_path1, str(k)), img)

            cur.execute("update  catalog.lp_Folder_Path_info set output_path ='{output_path}' where route='{route}'"
                        .format(output_path=output_path, route=route))
            conn.commit()

    except Exception as e:
        print(e)
        logger.error(msg='License_plate_route table  does not exist ' + str(e) + '\tTraceback\t' + '~'.join(
        str(traceback.format_exc()).split('\n')))



def detect(
        cfg,
        weights,
        imgpath,
        route,
        pro_count,
        # output_dir,  # output folder
        # undetected,
        img_size=416,
        conf_thres=0.09,
        nms_thres=0.8,
        save_txt= False,  # saving corrdinates in text file
        save_images=True
):

    try:

        conn = psycopg2.connect(host="10.1.1.168", database="License_Plate", user="postgres", password="postgres")
        cur = conn.cursor()

        conn1 = psycopg2.connect(host="10.1.1.168", database="ai_ml_hub", user="postgres", password="postgres")
        cur1 = conn1.cursor()
        # connection for ai-ml_hub db

    except Exception as e:
        cur.execute("Drop TABLE public.{route}".format(route=route))
        logger.error(msg='Unable to establish connection'+ str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    device = torch_utils.select_device()
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)  # delete output folder
    # os.makedirs(output_dir)  # make new output folder


    # Initialize model
    model = Darknet(cfg, img_size)

    try:
        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
            # print(model)
        else:  # darknet format
            load_darknet_weights(model, weights)

        model.to(device).eval()
        # print(eval)

    except Exception as e:
        cur.execute("Drop TABLE public.{route}".format(route=route))
        logger.error(msg='Model Weights Not Found\t' + str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    try:
        # Using mount path instead of windows path

        imgpath1 = imgpath.replace('%20', ' ').replace('\\', '/').replace('//', '/').replace(
            '/10.1.1.', '/mnt/vol1/10.1.1.')


        # Set Dataloader
        dataloader = LoadImages(imgpath1, img_size=img_size)
        # Get classes and colors
        classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
        # print(len(classes))
        # colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
        # colors = [[0, 51, 102] for _ in range(len(classes))]

        cur.execute("update  catalog.LP_Folder_Path_info set state='1' where route='{route}'".format(route=route))

    except Exception as e:
        cur.execute("Drop TABLE public.{route}".format(route=route))
        logger.error(msg='error Dataloader OR coco.data Not found\t' + str(e) + '\tTraceback\t' + '~'.join(
            str(traceback.format_exc()).split('\n')))

    try:
        for i, (path, img, im0) in enumerate(dataloader):

                # pro_count=0
                if i < pro_count:
                    continue
                # print(path)
                t = time.time()
                print('image %g/%g : ' % (i + 1, len(dataloader)), end='\r')

                # Get detections
                img = torch.from_numpy(img).unsqueeze(0).to(device)
                # print(img.shape)
                if ONNX_EXPORT:
                    torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
                    return
                pred = model(img)
                # print(pred)
                pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

                if len(pred) > 0:
                        # Run NMS on predictions

                            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
                            # Rescale boxes from 416 to true image size
                            detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

                            # Print results to screen
                            unique_classes = detections[:, -1].cpu().unique()
                            # for c in unique_classes:
                            #     n = (detections[:, -1].cpu() == c).sum()
                                # print('%g %ss' % (n, classes[int(c)]))

                            # Draw bounding boxes and labels of detections
                            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                                if save_txt:  # Write to file
                                    with open(save_path + '.txt', 'a') as file:
                                        file.write('%g %g %g %g %g %g\n' %
                                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                                roi_corners = np.array([[(int(x1 - 10), int(y1)), (int(x1 - 10), int(y2)),
                                                         (int(x2 + 10), int(y2)), (int(x2 + 10), int(y1))]],
                                                       dtype=np.int32)
                                # coord=to(roi_corners)
                                coordinate=roi_corners
                                coordinate=roi_corners.tolist()

                                # print(coordinate)

                                path_head_tail = os.path.split(path)
                                if int(x1) < 5:
                                    x1 = 0
                                    cord = [int(x1), int(y1), int(x2+5), int(y2)]

                                elif (int(x2)+5) > im0.shape[1]:
                                    x2= im0.shape[1]
                                    cord= [int(x1-5), int(y1), int(x2), int(y2)]

                                else:
                                    cord = [int(x1-5), int(y1), int(x2+5), int(y2)]

                                cur.execute(
                                    "insert into public.{route} values('{img_info}','{co}','T')"
                                    .format(img_info=path_head_tail[1], co=cord,
                                            route=route))

                                conn.commit()
                                # # print(roi_corners.tolist())
                                #
                                # # img= cv.imread('image.jpeg')
                                # blurred_image = cv.GaussianBlur(im0, (9, 9), 11)
                                # mask = np.zeros(im0.shape, dtype=np.uint8)
                                # channel_count = im0.shape[2]
                                # ignore_mask_color = (255,) * channel_count
                                # cv.fillPoly(mask, roi_corners, ignore_mask_color)
                                #
                                # # plt.imshow(mask)
                                # # plt.show()
                                # mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
                                # # plt.imshow(mask_inverse)
                                # # plt.show()
                                # im0 = cv.bitwise_and(blurred_image, mask) + cv.bitwise_and(im0, mask_inverse)
                                # # plt.imshow(im0)
                                # # plt.show()
                                # # plt.imshow(im0)
                                # # plt.show()

                dt = time.time() - t

                # print(' No detection (%.3fs)' % dt)

                cur.execute(
                    "update  catalog.lp_Folder_Path_info set processed_count={i} where route='{route}'"
                        .format(route=route, i=i+1))



                # if save_images:  # Save generated image with detections
                #     if len(pred) > 0:
                #         cv2.imwrite(save_path, im0)
                #      # print(save_path)
                #      # print(path)
                #
                #     else:
                #     # print(saved)
                #         save_path = str(Path(undetected) / Path(path).name)
                #         print(save_path)
                #         cv2.imwrite(save_path, im0)

    except Exception as e:
        cur.execute("Drop TABLE public.{route}".format(route=route))
        logger.error(msg='Query_Error' + str(e) + '\tTraceback\t' + '~'.join(str(traceback.format_exc()).split('\n')))

    finally:
        if (i+1) == len(dataloader):
            cur.execute("update  catalog.LP_Folder_Path_info set state='3' where route='{route}' and state =1"
                        .format(route=route))
            cur1.execute(
                "UPDATE catalog.gpu_server_info  SET proc_count = proc_count - 1 WHERE server_name='Server_3'")

            cur.execute("Delete from catalog.task_details where folder_path='{path}'".format(path=imgpath))

            conn.commit()
            conn.close()
            conn1.commit()
            conn1.close()

        else:
            # When path is interrupted
            cur.execute("UPDATE catalog.LP_Folder_Path_info SET state='2' WHERE route='{route}' and state = 1"
                        .format(route=route))

            ##wrapper use
            cur1.execute(
                "UPDATE catalog.gpu_server_info  SET proc_count = proc_count - 1 WHERE server_name='Server_3'")

            cur.execute("Delete from catalog.task_details where folder_path='{path}'".format(path=imgpath))

            conn.commit()
            conn.close()
            conn1.commit()
            conn1.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='/mnt/vol1/License-plate-detection/weights/weight.pt',
                        help='path to weights file')
    parser.add_argument('--imgpath', type=str, default='/mnt/vol1/License-plate-detection/test1', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--route', type=str, help='last part of directory')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    app.run(host='0.0.0.0', port=8000, debug=True)


