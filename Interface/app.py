import os
import gradio as gr
from PIL import Image
import rembg
import numpy as np

# Define paths (adjust these based on your repository layout)
OUTPUT_DIR = "out/demo_generation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(input_image):
    """
    Pipeline:
    1 & 2) Object detection & segmentation via rembg
    3) Add solid white background
    4) Resize to 224x224
    """
    if input_image is None:
        return None
    
    # Ensure image is a PIL Image
    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    else:
        img = input_image

    # Step 1 & 2: Remove background (Segments the primary detected object)
    # rembg automatically finds the dominant foreground object
    output_rgba = rembg.remove(img)

    # Step 3: Construct a solid white background image
    white_bg = Image.new("RGBA", output_rgba.size, (255, 255, 255, 255))
    # Alpha composite the segmented object over the white background
    final_img = Image.alpha_composite(white_bg, output_rgba).convert("RGB")

    # Step 4: Resize to 224x224 as required by the OccNet encoder
    resized_img = final_img.resize((224, 224), Image.Resampling.LANCZOS)
    
    return resized_img

def run_reconstruction(input_image):
    if input_image is None:
        return "Please upload an image first."
    
    # 1. Run the cleaning pipeline
    cleaned_img = preprocess_image(input_image)
    
    # 2. Save the cleaned image to a temporary path where generate.py can find it
    # For Occupancy Networks, it typically expects a specific dataset format 
    # or a direct path configured in your yaml.
    temp_img_path = "tmp_input.png"
    cleaned_img.save(temp_img_path)
    
    # 3. Call your model generation logic
    # Since generate.py is written as a script, we can import its execution loop 
    # or invoke it via system command. 
    # Given the CPU adjustment and batch size = 4 limits, running via terminal command is safest:
    config_path = "config/choy_single_cpu.yaml" # Your specific CPU config path
    
    try:
        # Command to run generation using your local configurations
        # We pass the input path if your generate.py was adapted to accept one,
        # otherwise ensure your config points to the directory containing 'tmp_input.png'
        os.system(f"python generate.py {config_path}")
        
        # 4. Locate the generated 3D mesh (.obj or .ply)
        # Occupancy Networks stores outputs under out/.../generation/
        # Find the latest mesh file generated
        generated_mesh_path = None
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file.endswith('.obj') or file.endswith('.ply'):
                    generated_mesh_path = os.path.join(root, file)
                    break
        
        if generated_mesh_path and os.path.exists(generated_mesh_path):
            return cleaned_img, generated_mesh_path
        else:
            return cleaned_img, None
            
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return cleaned_img, None

# --- Gradio UI Design ---
with gr.Blocks() as demo:
    gr.Markdown("# 3D Object Reconstruction via Occupancy Networks")
    gr.Markdown("Upload an image of an object. The app will automatically isolate it, format it, and reconstruct its 3D structure on the CPU.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Reconstruct Object")
            
        with gr.Column():
            cleaned_preview = gr.Image(label="Preprocessed Input (224x224 White BG)")
            output_3d = gr.Model3D(label="Reconstructed 3D Mesh")

    submit_btn.click(
        fn=run_reconstruction,
        inputs=[input_img],
        outputs=[cleaned_preview, output_3d]
    )

demo.launch()