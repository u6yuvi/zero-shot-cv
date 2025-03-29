import gradio as gr
from model_ops import YOLOEModel, encode_image
import tempfile
import os
import numpy as np
import cv2
from PIL import Image

# Initialize the model
model = YOLOEModel()

def add_class(new_class, class_list):
    """Add a new class to the available classes"""
    if new_class and new_class not in class_list:
        class_list.append(new_class)
    return gr.Dropdown(choices=class_list, value=new_class if new_class else class_list[0]), class_list

def create_bbox(image, evt: gr.SelectData, selected_class):
    """Handle bounding box creation"""
    if not hasattr(create_bbox, 'boxes'):
        create_bbox.boxes = []
        create_bbox.start_pos = None
    
    # Get the coordinates from the selection event
    x, y = evt.index
    
    # Store start position or create box
    if create_bbox.start_pos is None:
        # First click - store start position
        create_bbox.start_pos = (x, y)
        return image
    else:
        # Second click - create box
        x1, y1 = create_bbox.start_pos
        x2, y2 = x, y
        create_bbox.start_pos = None  # Reset for next box
        
        # Calculate width and height
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Ensure x1,y1 is the top-left corner
        x = min(x1, x2)
        y = min(y1, y2)
        
        # Add the box to the model with selected class
        model.add_box(x, y, width, height, selected_class)
        create_bbox.boxes.append({
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'label': selected_class
        })
        
        # Create visualization
        boxes = [[box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']] 
                 for box in create_bbox.boxes]
        labels = [box['label'] for box in create_bbox.boxes]
        
        # Convert image to numpy array for visualization
        if isinstance(image, np.ndarray):
            img_array = image.copy()
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
        
        # Convert numpy array to PIL Image if needed
        if isinstance(source_image, np.ndarray):
            source_image = Image.fromarray(source_image)
        if isinstance(target_image, np.ndarray):
            target_image = Image.fromarray(target_image)
        
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
    gr.Markdown("""
    ### Instructions:
    1. Add new classes using the text input below
    2. Select a class from the dropdown menu
    3. Create bounding boxes:
       - First click: Set the starting corner
       - Second click: Set the ending corner
    4. Repeat steps 2-3 for more boxes
    5. Use the clear button to remove all boxes
    6. Upload a target image and run prediction
    """)
    
    # Store class list in state
    class_list = gr.State(["eye", "cat", "dog", "person", "car"])
    
    with gr.Row():
        # Add new class input
        new_class_input = gr.Textbox(label="Add New Class")
        add_class_btn = gr.Button("Add Class")
    
    with gr.Row():
        with gr.Column():
            source_image = gr.Image(
                label="Source Image",
                type="numpy",
                interactive=True,
                height=400,
                tool="bbox"  # Use bbox tool for bounding box creation
            )
            target_image = gr.Image(
                label="Target Image",
                type="pil",
                height=400
            )
            
        with gr.Column():
            output_image = gr.Image(label="Output", height=400)
    
    with gr.Row():
        # Add class selection dropdown
        class_dropdown = gr.Dropdown(
            choices=class_list.value,
            value=class_list.value[0] if class_list.value else None,
            label="Select Class"
        )
        
        # Add clear button
        clear_btn = gr.Button("Clear Boxes")
        
        # Add predict button
        predict_btn = gr.Button("Run Prediction")
    
    # Handle adding new class
    add_class_btn.click(
        add_class,
        inputs=[new_class_input, class_list],
        outputs=[class_dropdown, class_list]
    )
    
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