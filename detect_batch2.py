from celery_config import app
from detect_api2 import detect
from detect_api2 import blur
@app.task(bind=True)
def detect_batch1(self, imgpath, route, pro_count):
    detect(
         './cfg/yolov3.cfg',
         '/mnt/vol1/License-plate-detection/weights/weight.pt',
        imgpath,
        route,
        pro_count
        # conf_thres=thres
    )


@app.task(bind=True)
def detect_blur1(self, imgpath, output_dir, range_start, range_end):
    blur(imgpath, output_dir, range_start, range_end)