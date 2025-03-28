import gradio as gr
from model_ops import YOLOEModel, encode_image
import tempfile
import os
import numpy as np
import cv2

# Initialize the model
model = YOLOEModel()

def create_bbox(image, evt: gr.SelectData, selected_class):
    """Handle bounding box creation"""
    if not hasattr(create_bbox, 'boxes'):
        create_bbox.boxes = []
    
    # Get the coordinates from the selection
    x, y = evt.index  # Current point
    
    # Add the box to the model with selected class
    # For simplicity, create a fixed-size box around the clicked point
    box_size = 100  # Default box size
    x = max(0, x - box_size//2)
    y = max(0, y - box_size//2)
    
    model.add_box(x, y, box_size, box_size, selected_class)
    create_bbox.boxes.append({
        'x': x,
        'y': y,
        'width': box_size,
        'height': box_size,
        'label': selected_class
    })
    
    # Create visualization
    boxes = [[box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']] 
             for box in create_bbox.boxes]
    labels = [box['label'] for box in create_bbox.boxes]
    
    # Convert image to numpy array for visualization
    if isinstance(image, np.ndarray):
        img_array = image.copy()  # Create a copy to avoid modifying the original
    else:
        img_array = np.array(image)
    
    # Draw boxes on the image
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        # Draw rectangle
        img_array = cv2.rectangle(img_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Add label
        img_array = cv2.putText(img_array, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_array

def clear_boxes():
    """Clear all bounding boxes"""
    if hasattr(create_bbox, 'boxes'):
        create_bbox.boxes = []
    model.boxes = []
    model.names = []
    return None

def predict(source_image, target_image):
    """Run prediction with the created bounding boxes"""
    if source_image is None or target_image is None:
        return None
        
    # Save the uploaded images temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as source_temp, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as target_temp:
        
        # Save images
        source_image.save(source_temp.name)
        target_image.save(target_temp.name)
        source_path = source_temp.name
        target_path = target_temp.name
        
        # Run prediction
        results = model.predict(source_path, target_path)
        
        # Get visualization results
        if results is not None:
            output = model.visualize_results(target_path, results)
        else:
            output = None
        
        # Clean up temporary files
        os.unlink(source_path)
        os.unlink(target_path)
        
        return output

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# YOLOE Interactive Bounding Box Demo")
    gr.Markdown("Click on the image to add bounding boxes. Each click will create a box centered at the click position.")
    
    with gr.Row():
        with gr.Column():
            source_image = gr.Image(label="Source Image", type="pil", interactive=True, height=400)
            target_image = gr.Image(label="Target Image", type="pil", height=400)
            
        with gr.Column():
            output_image = gr.Image(label="Output", height=400)
    
    with gr.Row():
        # Add class selection dropdown
        class_dropdown = gr.Dropdown(
            choices=model.available_classes,
            value=model.available_classes[0],  # Default to first class
            label="Select Class"
        )
        
        # Add box size slider
        box_size_slider = gr.Slider(
            minimum=20,
            maximum=200,
            value=100,
            step=10,
            label="Box Size"
        )
        
        # Add clear button
        clear_btn = gr.Button("Clear Boxes")
        
        # Add predict button
        predict_btn = gr.Button("Run Prediction")
    
    # Handle bounding box creation
    source_image.select(
        create_bbox,
        inputs=[source_image, class_dropdown],
        outputs=source_image
    )
    
    # Handle clear boxes
    clear_btn.click(
        clear_boxes,
        outputs=source_image
    )
    
    # Handle prediction
    predict_btn.click(
        predict,
        inputs=[source_image, target_image],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch() 