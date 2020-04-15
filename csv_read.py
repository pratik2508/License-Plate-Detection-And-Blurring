import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
import time

def annot_format():

    try:

        txt_file_list = []

        for k, i in zip(path, annot_path):
            time1 = time.time()
            # DataFrame Preprocessing
            df = pd.read_csv(i)
            df.drop(['file_size', 'region_count', 'file_attributes', 'region_id', 'region_attributes'], inplace=True,
                     axis=1)

            df.drop(df[df['region_shape_attributes'] == '{}'].index, inplace=True)


            dir_list = []
            img_list = []
            img_list.append(os.listdir(k))
            img_list[0].sort()

            img_path_train = k + '_train'  # training path created
            if os.path.exists(img_path_train):
                print('folder already exist:-', img_path_train)
                # os.system("rm -rf _" + img_path_train)
                shutil.rmtree(img_path_train)
                os.mkdir(img_path_train)

            else:
                os.mkdir(img_path_train)

            # for imgpath in img_list[0]:
            #     final_img_list.append(imgpath)  # making images path to training dir path

            # final_img_list.sort()

            for iter, (p, q) in enumerate(zip (df['filename'], df['region_shape_attributes'])):
                z = eval(q)             # q is a string type var
                x = z["x"]              # eval act as decorater that is used to pick key values pair in str object
                y = z["y"]
                w = z["width"]
                h = z["height"]

                for iter1, file in enumerate(img_list[0]):
                # if p in img_list[0]:
                    if file == p:
                        # file=p
                        txt_file = file[:-4] + ".txt"  # txt file created
                        txt_file_path = os.path.join(img_path_train, txt_file)

                        file_path = os.path.join(k, file)  # for reading images

                        img = plt.imread(file_path)
                        height = img.shape[0]
                        width = img.shape[1]

                        # appending all txt files of a folder to a list that will be returned
                        dir_list.append(txt_file_path)

                        # print(iter1, (x, y, w, h))
                        x1 = x + (w / 2)  # x centre coordinates of the bounding box
                        y1 = y + (h / 2)  # y centre coordinates of the bounding box

                        # final needed coordinates as per coco data
                        x_center = round(x1 / width, 6)
                        y_center = round(y1 / height, 6)
                        final_width = round(w / width, 6)
                        final_height = round(h / height, 6)
                        # print(x_center, y_center, final_width, final_height)

                        shutil.copy(file_path, img_path_train)  # copying images to training folder

                        # if df.iloc[iter]['filename'] == df.iloc[iter]

                        # writing corresponding txt file to training folder
                        writefile = open(txt_file_path, "a+")
                        writefile.write("%d %s %s %s %s\n" % (0, x_center, y_center, final_width, final_height))
                        writefile.close()

            for i in dir_list:  # making final list of all text file which is txt_file_list
                txt_file_list.append(i)
            print(len(txt_file_list))
        print("time",time.time()-time1)
        print("csv to txt format conversion complete")

    except Exception as e:
        print(e)

    return txt_file_list



if __name__=="__main__":

    path = ['/mnt/vol1/06022018/L', '/mnt/vol1/06022018/R', '/mnt/vol1/06022018/F']

    annot_path =['/mnt/vol1/license_plate_detection_anno/L_FINAL.csv',
                 '/mnt/vol1/license_plate_detection_anno/R_FINAL.csv',
                 '/mnt/vol1/license_plate_detection_anno/F_FINAL.csv']


    print("csv to txt format conversion Started")
    all_txt_file_list = annot_format()
    all_txt_file_list.sort()


    train, test = train_test_split(all_txt_file_list, test_size=0.2, shuffle=False, random_state=25)
    print(len(train), len(test))

    train_file = os.path.join('/mnt/vol1/06022018', 'train.txt')
    test_file = os.path.join('/mnt/vol1/06022018', 'test.txt')
    print(train_file)
    print(test_file)

    writefile = open(train_file, 'w')
    for iter in train:
        writefile.write(os.path.join('/mnt/vol1/06022018', iter))
        writefile.write('\n')
    writefile.close()

    writefile1 = open(test_file, 'w')
    for iter1 in test:
        writefile1.write(os.path.join('/mnt/vol1/06022018', iter1))
        writefile1.write('\n')
    writefile1.close()






