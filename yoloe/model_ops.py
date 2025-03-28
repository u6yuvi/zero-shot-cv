import supervision as sv
from ultralytics import YOLOE
from PIL import Image
import numpy as np
import base64
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded

class YOLOEModel:
    def __init__(self, model_path="yoloe-v8l-seg.pt"):
        self.model = YOLOE(model_path)
        self.names = []
        self.boxes = []
        self.available_classes = ["eye", "cat", "dog", "person", "car"]  # Add more classes as needed
        
    def set_classes(self, names):
        self.names = names
        self.model.set_classes(names, self.model.get_text_pe(names))
        
    def add_box(self, x, y, width, height, label):
        if label not in self.names:
            self.names.append(label)
            self.model.set_classes(self.names, self.model.get_text_pe(self.names))
        
        self.boxes.append({
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'label': label
        })
        
    def get_bboxes_array(self):
        if not self.boxes:
            return None, None
            
        bboxes = np.array([
            [
                box['x'],
                box['y'],
                box['x'] + box['width'],
                box['y'] + box['height']
            ] for box in self.boxes
        ], dtype=np.float64)
        
        cls = np.array([self.names.index(box['label']) for box in self.boxes], dtype=np.int32)
        
        return bboxes, cls
        
    def predict(self, source_image_path, target_image_path):
        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)
        
        bboxes, cls = self.get_bboxes_array()
        if bboxes is None:
            return None
            
        prompts = dict(bboxes=bboxes, cls=cls)
        
        # First prediction to get VPE
        results = self.model.predict(source_image, prompts=prompts, predictor=YOLOEVPSegPredictor, return_vpe=True)
        if hasattr(results[0], 'vpe'):
            self.model.set_classes(self.names, results[0].vpe)
        else:
            # If VPE is not available, use text embeddings
            self.model.set_classes(self.names, self.model.get_text_pe(self.names))
        
        # Reset predictor
        self.model.predictor = None
        
        # Final prediction on target image
        results = self.model.predict(target_image)
        return results[0]
        
    def visualize_results(self, image_path, results):
        if results is None:
            return None
            
        image = Image.open(image_path)
        detections = sv.Detections.from_ultralytics(results)
        
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
        
        return annotated_image 