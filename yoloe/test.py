import supervision as sv
from ultralytics import YOLOE
from PIL import Image
import numpy as np

IMAGE_PATH = "/Users/uv/Documents/work/gitrepos/zero-shot-cv/yoloe/index_images/cat/cat1.jpg"
NAMES = ["cat", "eye"]

model = YOLOE("yoloe-v8l-seg.pt")#.cuda()
model.set_classes(NAMES, model.get_text_pe(NAMES))

image = Image.open(IMAGE_PATH)
results = model.predict(image, conf=0.1, verbose=False)

detections = sv.Detections.from_ultralytics(results[0])

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

annotated_image


import base64

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded


SOURCE_IMAGE_PATH = "/Users/uv/Documents/work/gitrepos/zero-shot-cv/yoloe/index_images/cat/cat1.jpg"
TARGET_IMAGE_PATH = "/Users/uv/Documents/work/gitrepos/zero-shot-cv/yoloe/index_images/cat/cat1.jpg"
NAMES = ['eye']


boxes = [{'x': 1695, 'y': 1917, 'width': 425, 'height': 420, 'label': 'eye'},
 {'x': 953, 'y': 1562, 'width': 398, 'height': 507, 'label': 'eye'}]

bboxes = np.array([
    [
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ] for box in boxes
], dtype=np.float64)

cls = np.array([NAMES.index(box['label']) for box in boxes], dtype=np.int32)

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

model = YOLOE("yoloe-v8l-seg.pt")#.cuda()
prompts = dict(bboxes=bboxes, cls=cls)

source_image = Image.open(SOURCE_IMAGE_PATH)
target_image = Image.open(TARGET_IMAGE_PATH)

model.predict(source_image, prompts=prompts, predictor=YOLOEVPSegPredictor, return_vpe=True)
model.set_classes(NAMES, model.predictor.vpe)
model.predictor = None

results = model.predict(target_image)
detections = sv.Detections.from_ultralytics(results[0])

annotated_image = target_image.copy()
annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

print(results[0].boxes)