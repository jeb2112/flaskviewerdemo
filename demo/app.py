import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.path import Path
import nibabel as nib
from flask import Flask, jsonify, request, session, Response, send_file
from flask_cors import CORS
import io
import base64
from PIL import Image
import threading
from sklearn.cluster import KMeans
from threading import Thread
from queue import Queue
import time
import scipy.ndimage
from scipy.ndimage import binary_dilation, binary_fill_holes

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--datadir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom2nifti_upload/DSC_0010/20240422")

args = parser.parse_args()

app = Flask(__name__, static_folder="static")
CORS(app)
app.secret_key = 'test'

# Global variables to store image data
current_images = {
    'image1': None,
    'image2': None,
    'image1_normalized': None,
    'image2_normalized': None,
    'current_slice': 0,
    'num_slices': 0,
    'threshold': 1.0,
    'mask': None,  # Current mask being drawn
    'saved_masks': [],  # List of saved masks
    'undo_stack': [],  # Stack for undo operations
    'neighbor_mode': False,  # Flag for neighbor selection mode
    'lock': threading.Lock(),  # Add thread lock
    'last_click_x': None,  # Store last clicked x coordinate
    'last_click_y': None,   # Store last clicked y coordinate
    'transform_left': None,  # Store transform for left image
    'transform_right': None,  # Store transform for right image
    'click_coordinates': [],  # List to store all click coordinates
    'norm_params': None  # Storage for per-slice normalization parameters
}

# Add after other global variables
processing_status = {
    'is_processing': False,
    'progress': 0,
    'error': None,
    'complete': False
}

def process_images_async():
    """Background thread for image processing"""
    global processing_status
    try:
        processing_status['is_processing'] = True
        processing_status['progress'] = 0
        processing_status['error'] = None
        
        # Print some statistics about the input images
        print(f"Image1 stats - min: {np.min(current_images['image1'])}, max: {np.max(current_images['image1'])}, mean: {np.mean(current_images['image1'])}", flush=True)
        print(f"Image2 stats - min: {np.min(current_images['image2'])}, max: {np.max(current_images['image2'])}, mean: {np.mean(current_images['image2'])}", flush=True)
        
        # Create mask for entire volume
        mask = (current_images['image1'] > 0) & (current_images['image2'] > 0)
        
        # Downsample the mask to reduce computation time
        downsample_factor = 4
        mask_downsampled = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
        img1_downsampled = current_images['image1'][::downsample_factor, ::downsample_factor, ::downsample_factor]
        img2_downsampled = current_images['image2'][::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        # Create points for clustering
        valid_points = np.column_stack((
            img1_downsampled[mask_downsampled].flatten(),
            img2_downsampled[mask_downsampled].flatten()
        ))
        
        if len(valid_points) > 0:  # Only process if we have valid points
            # Simple normalization for K-means stability
            valid_points = (valid_points - np.mean(valid_points, axis=0)) / (np.std(valid_points, axis=0) + 1e-6)
            
            # Run K-means
            kmeans = KMeans(n_clusters=2, n_init=1, max_iter=100)
            kmeans.fit(valid_points)
            
            # Identify background cluster
            cluster_centers = kmeans.cluster_centers_
            distances = np.sum(cluster_centers**2, axis=1)
            background_cluster = np.argmin(distances)
            
            # Get full resolution points
            valid_points_full = np.column_stack((
                current_images['image1'][mask].flatten(),
                current_images['image2'][mask].flatten()
            ))
            
            # Normalize points and find cluster assignments
            valid_points_normalized = (valid_points_full - np.mean(valid_points_full, axis=0)) / (np.std(valid_points_full, axis=0) + 1e-6)
            distances_to_centers = np.sum((valid_points_normalized[:, np.newaxis] - cluster_centers)**2, axis=2)
            labels_full = np.argmin(distances_to_centers, axis=1)
            
            # Calculate normalization parameters for background points
            background_points = valid_points_full[labels_full == background_cluster]
            if len(background_points) > 0:
                means = np.mean(background_points, axis=0)
                stds = np.std(background_points, axis=0)
                
                # Initialize normalized images
                with current_images['lock']:
                    current_images['image1_normalized'] = (current_images['image1'] - means[0]) / (stds[0] + 1e-6)
                    current_images['image2_normalized'] = (current_images['image2'] - means[1]) / (stds[1] + 1e-6)
                    
                    # Set background to zero
                    current_images['image1_normalized'][~mask] = 0
                    current_images['image2_normalized'][~mask] = 0
                    
                    print("\nVolume normalization complete", flush=True)
        
        processing_status['progress'] = 100
        processing_status['complete'] = True
            
    except Exception as e:
        import traceback
        print(f"Error during normalization: {str(e)}", flush=True)
        print("Traceback:", flush=True)
        print(traceback.format_exc(), flush=True)
        processing_status['error'] = str(e)
    finally:
        processing_status['is_processing'] = False

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route('/load_images', methods=['POST'])
def load_images():
    print("Received load_images request", flush=True)
    data = request.get_json()
    print("Request data:", data, flush=True)
    image1_path = data.get('image1_path')
    image2_path = data.get('image2_path')
    
    if not image1_path or not image2_path:
        return jsonify({"error": "Missing image paths"}), 400
        
    try:
        # Load NIfTI images
        full_path1 = os.path.join(args.datadir, image1_path)
        full_path2 = os.path.join(args.datadir, image2_path)
        
        img1 = nib.load(full_path1)
        img2 = nib.load(full_path2)
        print("Images loaded successfully", flush=True)
        
        # Store image data and rotate by 90 degrees
        with current_images['lock']:
            # Rotate images by 90 degrees using np.rot90
            current_images['image1'] = np.rot90(np.array(img1.dataobj), k=-1)
            current_images['image2'] = np.rot90(np.array(img2.dataobj), k=-1)
            current_images['num_slices'] = current_images['image1'].shape[2]
            current_images['current_slice'] = 25
            current_images['mask'] = np.zeros_like(current_images['image1'], dtype=bool)
        
        # Reset processing status
        processing_status['is_processing'] = False
        processing_status['progress'] = 0
        processing_status['error'] = None
        processing_status['complete'] = False
        
        # Start background processing
        thread = Thread(target=process_images_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Images loading started",
            "num_slices": current_images['num_slices'],
            "current_slice": current_images['current_slice']
        })
    except Exception as e:
        print(f"Error in load_images: {str(e)}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route('/normalization_status', methods=['GET'])
def get_normalization_status():
    """Check the status of image normalization"""
    return jsonify({
        'is_processing': processing_status['is_processing'],
        'progress': processing_status['progress'],
        'error': processing_status['error'],
        'complete': processing_status['complete']
    })

@app.route('/get_slice', methods=['GET'])
def get_slice():
    print("Received get_slice request", flush=True)
    if current_images['image1'] is None or current_images['image2'] is None:
        print("No images loaded", flush=True)
        return jsonify({"error": "No images loaded"}), 400
        
    slice_idx = request.args.get('slice', type=int, default=current_images['current_slice'])
    print(f"Requested slice: {slice_idx}", flush=True)
    
    if slice_idx < 0 or slice_idx >= current_images['num_slices']:
        print(f"Invalid slice index: {slice_idx}", flush=True)
        return jsonify({"error": "Invalid slice index"}), 400
    
    # Use thread lock for matplotlib operations
    with current_images['lock']:
        current_images['current_slice'] = slice_idx
        
        # Get image dimensions
        height, width = current_images['image1_normalized'].shape[:2]
        
        # Set scale factor for display
        scale_factor = 2.0
        scaled_width = int(width * scale_factor)
        scaled_height = int(height * scale_factor)
        
        # Create figure at scaled dimensions
        dpi = 100
        figsize = (scaled_width * 2 / dpi, scaled_height / dpi)  # width*2 for two images side by side
        print(f"Creating figure with size {figsize} inches at {dpi} DPI (scale factor: {scale_factor})", flush=True)
        
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Set axes to exactly match scaled dimensions
        ax1.set_position([0, 0, 0.5, 1.0])  # Left half
        ax2.set_position([0.5, 0, 0.5, 1.0])  # Right half
        
        # Display normalized images with upper origin to match screen coordinates
        ax1.imshow(current_images['image1_normalized'][:, :, slice_idx], 
                  cmap='gray',
                  interpolation='none',
                  origin='upper')
        ax2.imshow(current_images['image2_normalized'][:, :, slice_idx], 
                  cmap='gray',
                  interpolation='none',
                  origin='upper')
        
        # Set the axes limits explicitly
        ax1.set_xlim(0, width)
        ax1.set_ylim(height, 0)  # Note: reversed y limits for upper origin
        ax2.set_xlim(0, width)
        ax2.set_ylim(height, 0)  # Note: reversed y limits for upper origin
        
        # Store the transforms for coordinate conversion
        current_images['transform_left'] = ax1.transData + ax1.transAxes.inverted()
        current_images['transform_right'] = ax2.transData + ax2.transAxes.inverted()
        
        # Add mask overlay if it exists
        if current_images['mask'] is not None:
            mask_slice = current_images['mask'][:, :, slice_idx]
            if np.any(mask_slice):
                mask_overlay = np.zeros((*mask_slice.shape, 4))
                mask_overlay[mask_slice, 0] = 1  # Red channel
                mask_overlay[mask_slice, 3] = 0.5  # Alpha channel
                ax1.imshow(mask_overlay, interpolation='none', origin='upper')
                ax2.imshow(mask_overlay, interpolation='none', origin='upper')
        
        ax1.axis('off')
        ax2.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
        
        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            "image": image_base64,
            "current_slice": slice_idx,
            "dimensions": {
                "width": scaled_width * 2,  # Total width of both images
                "height": scaled_height,     # Height of images
                "single_width": scaled_width,  # Width of a single image
                "single_height": scaled_height,  # Height of a single image
                "scale_factor": scale_factor,
                "original_width": width,     # Original image width
                "original_height": height    # Original image height
            }
        })

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    data = request.get_json()
    threshold = data.get('threshold', current_images['threshold'])
    
    if threshold < 0.1 or threshold > 6.0:
        return jsonify({"error": "Invalid threshold value"}), 400
    
    with current_images['lock']:
        current_images['threshold'] = threshold
        
        # If we have click coordinates, recalculate the mask for all clicks
        if current_images['click_coordinates']:
            print(f"Recalculating mask with new threshold {threshold} for all clicks")
            
            # Clear the entire mask
            current_images['mask'] = np.zeros_like(current_images['image1'], dtype=bool)
            
            # Recalculate mask for each click
            for x, y in current_images['click_coordinates']:
                if current_images['neighbor_mode']:
                    new_mask = create_neighbor_mask(x, y)
                    # For neighbor mode, we still only update the current slice
                    slice_idx = current_images['current_slice']
                    current_images['mask'][:, :, slice_idx] |= new_mask
                else:
                    # For threshold mode, we get the full 3D mask
                    new_mask = create_threshold_mask(x, y)
                    current_images['mask'] |= new_mask
            
            print(f"Mask updated with new threshold for {len(current_images['click_coordinates'])} clicks")
    
    return jsonify({"message": "Threshold updated successfully"})

@app.route('/add_to_mask', methods=['POST'])
def add_to_mask():
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    
    if x is None or y is None:
        return jsonify({"error": "Missing coordinates"}), 400
        
    print(f"Received click at ({x}, {y})", flush=True)
    
    # Store the click coordinates
    current_images['click_coordinates'].append((x, y))
    current_images['last_click_x'] = x
    current_images['last_click_y'] = y
    
    # Save current mask state for undo
    current_images['undo_stack'].append(current_images['mask'].copy())
    
    # Create new mask
    if current_images['neighbor_mode']:
        new_mask = create_neighbor_mask(x, y)
        # For neighbor mode, we still only update the current slice
        slice_idx = current_images['current_slice']
        current_images['mask'][:, :, slice_idx] |= new_mask
    else:
        # For threshold mode, we get the full 3D mask
        new_mask = create_threshold_mask(x, y)
        current_images['mask'] |= new_mask
    
    return jsonify({"message": "Mask updated successfully"})

@app.route('/undo', methods=['POST'])
def undo():
    """Undo the last mask operation"""
    try:
        with current_images['lock']:
            if not current_images['undo_stack']:
                return jsonify({"error": "Nothing to undo"}), 400
                
            # Restore the previous mask state
            current_images['mask'] = current_images['undo_stack'].pop()
            
            # If we're undoing a click, remove the last click coordinates
            if current_images['click_coordinates']:
                current_images['click_coordinates'].pop()
                if current_images['click_coordinates']:
                    current_images['last_click_x'], current_images['last_click_y'] = current_images['click_coordinates'][-1]
                else:
                    current_images['last_click_x'] = None
                    current_images['last_click_y'] = None
            
            print("Undo successful - restored previous mask state")
            return jsonify({"message": "Undo successful"})
    except Exception as e:
        print("Error in undo:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/toggle_neighbor_mode', methods=['POST'])
def toggle_neighbor_mode():
    with current_images['lock']:
        current_images['neighbor_mode'] = not current_images['neighbor_mode']
    return jsonify({
        "message": "Neighbor mode toggled",
        "neighbor_mode": current_images['neighbor_mode']
    })

@app.route('/save_mask', methods=['POST'])
def save_mask():
    data = request.get_json()
    mask_name = data.get('name', 'mask.nii.gz')
    
    with current_images['lock']:
        if current_images['mask'] is None:
            return jsonify({"error": "No mask to save"}), 400
            
        try:
            # Rotate mask back to original orientation before saving
            mask_to_save = np.rot90(current_images['mask'], k=1)
            
            # Create NIfTI image from mask
            mask_nifti = nib.Nifti1Image(mask_to_save.astype(np.uint8), np.eye(4))
            # Save to file
            output_path = os.path.join(args.datadir, mask_name)
            nib.save(mask_nifti, output_path)
            return jsonify({"message": f"Mask saved as {mask_name}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/clear_mask', methods=['POST'])
def clear_mask():
    """Clear the current mask"""
    with current_images['lock']:
        current_images['mask'] = np.zeros_like(current_images['image1'], dtype=bool)
        current_images['click_coordinates'] = []  # Clear stored click coordinates
        current_images['last_click_x'] = None
        current_images['last_click_y'] = None
    return jsonify({"message": "Mask cleared successfully"})

def fill_mask(mask):
    """Fill holes in the mask"""
    # Dilate mask with 3D structure element
    strel = np.ones((3,3,3))  # 3D structure element
    dilated_mask = binary_dilation(mask, strel)
    return binary_fill_holes(dilated_mask)

def create_threshold_mask(x, y):
    """Create mask based on intensity threshold"""
    slice_idx = current_images['current_slice']
    threshold = current_images['threshold']
    
    # Get image dimensions
    height, width = current_images['image1_normalized'].shape[:2]
    num_slices = current_images['image1_normalized'].shape[2]
    
    # Scale factor used in display
    scale_factor = 2.0
    scaled_width = int(width * scale_factor)  # Width in display space
    
    # Determine if click was on right image
    is_right_image = x >= scaled_width
    if is_right_image:
        x -= scaled_width
    
    # Convert display coordinates to data coordinates
    img_x = int(x / scale_factor)
    img_y = int(y / scale_factor)
    
    # Ensure coordinates are within bounds
    if not (0 <= img_x < width and 0 <= img_y < height):
        return np.zeros((height, width, num_slices), dtype=bool)
    
    # Get reference values at clicked point
    val1 = current_images['image1_normalized'][:, :, slice_idx][img_y, img_x]
    val2 = current_images['image2_normalized'][:, :, slice_idx][img_y, img_x]
    
    # Create differences for the entire volume
    diff1 = np.abs(current_images['image1_normalized'] - val1)
    diff2 = np.abs(current_images['image2_normalized'] - val2)
    
    # Create initial 3D mask
    initial_volume_mask = (diff1 <= threshold) & (diff2 <= threshold)
    
    # Find connected components in 3D
    labeled_array, num_features = scipy.ndimage.label(initial_volume_mask, structure=np.ones((3,3,3)))
    clicked_label = labeled_array[img_y, img_x, slice_idx]
    
    if clicked_label == 0:
        return np.zeros((height, width, num_slices), dtype=bool)
    
    # Create final 3D mask with only the clicked component
    volume_mask = labeled_array == clicked_label
    
    # Fill holes in the entire 3D mask
    volume_mask = fill_mask(volume_mask)
    
    return volume_mask

def create_neighbor_mask(x, y):
    """Create mask including neighboring points"""
    # Get image dimensions
    height, width = current_images['image1_normalized'].shape[:2]
    
    # Create mask
    mask = np.zeros((height, width), dtype=bool)
    
    # Get neighbors in 3x3 grid
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbor_mask = create_threshold_mask(nx, ny)
                mask |= neighbor_mask
    
    return mask

if __name__ == "__main__":
    # Print registered routes
    print("\nRegistered Routes:", flush=True)
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.methods} - {rule}", flush=True)
    
    print("\nStarting Flask server...", flush=True)
    # Disable Flask's reloader to avoid matplotlib issues
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
