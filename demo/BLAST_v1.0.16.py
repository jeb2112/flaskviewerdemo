# BLAST UI for RAD NEC DL Project
# Chris Heyn 2024

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import label, binary_fill_holes, binary_closing, binary_dilation
import copy

class MRIViewer(matplotlib.figure.Figure):
    def __init__(self, filepath, image1_filename, image2_filename, figsize=(8, 5), **kwargs):
        super().__init__(figsize=figsize, **kwargs)

        # Store filepaths
        self.filepath = filepath
        self.image1_filename = image1_filename
        self.image2_filename = image2_filename
        
        # Set the figure and axes
        self.patch.set_facecolor('black')
        self.fig, self.ax1, self.ax2 = self.create_axes()
        
        # Load MRI data
        self.image1_data = self.load_images(self.filepath, self.image1_filename)
        self.image2_data = self.load_images(self.filepath, self.image2_filename)
        self.slice = self.image1_data.shape[2] // 2
        self.num_slices = self.image1_data.shape[2]

        # Returns a dictionary of centroid and std image1 and image2 for the dominant cluster in each slice
        self.centroid_std_dict = self.normalize(self.image1_data, self.image2_data)

        # Initial display and caching of imshow objects
        self.img1 = self.ax1.imshow(self.image1_data[:, :, self.slice], cmap='gray', interpolation='none')
        self.ax1.axis('off')
        self.img2 = self.ax2.imshow(self.image2_data[:, :, self.slice], cmap='gray', interpolation='none')
        self.ax2.axis('off')

        # Slider setup
        self.ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(self.ax_slider, 'Slice', 0, self.num_slices - 1, valinit=self.slice, valstep=1)
        self.slider.label.set_color('white')
        self.slider.valtext.set_color('white')
        self.slider.on_changed(self.slider_changed)

        # Mouse clicks
        self.leftclick = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_image_click)   

        # Scroller setup
        self.adjustable_threshold = 0.4
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Connect key press
        self.neighbor_select = False
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initializes variable and arrays
        self.undo_saved_connected_mask = np.zeros_like(self.image1_data, dtype=np.uint8).astype(bool)
        self.saved_connected_mask = np.zeros_like(self.image1_data, dtype=np.uint8).astype(bool)
        self.H, self.W = self.saved_connected_mask.shape[:2]  # Get dimensions of image

        plt.show() 

    def create_axes(self):
        fig, axes = plt.subplots(1, 2, figsize=self.get_size_inches())
        fig.patch.set_facecolor('black')
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0)
        return fig, axes[0], axes[1]

    def load_images(self, filepath, image_name):
        img = nib.load(filepath + image_name)
        nii = img.get_fdata()
        #nii = nib.as_closest_canonical(img).get_fdata()
        nii = np.rot90(nii, k=-1)
        return nii
    
    def save_mask(self):
        
        # Convert numpy array to nii image
        mask = self.filled_mask.astype(np.uint8)
        mask = np.rot90(mask, k=1)
        affine = np.eye(4)
        nii = nib.Nifti1Image(mask, affine)
       
        # User inputs filename
        saved_name = input("Enter name of mask to save:") or "mask_TC.nii"

        # Save the NIfTI file
        nib.save(nii, self.filepath + saved_name)
        print('Nifti mask saved')
    
    def normalize(self, image1_data, image2_data):
        centroid_std_dict = {}
        for slice in range(self.num_slices):
            image1_slice = image1_data[:, :, slice].flatten()
            image2_slice = image2_data[:, :, slice].flatten()
            points = np.column_stack((image1_slice, image2_slice))

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=2, n_init = 'auto', random_state=0).fit(points)
            centroids = kmeans.cluster_centers_
            # Ensure cluster 2 is the one closest to (0, 0)
            distances_to_origin = np.linalg.norm(centroids, axis=1)
            sorted_indices = np.argsort(distances_to_origin)
            labels = kmeans.labels_
            cluster_1_idx, cluster_2_idx = sorted_indices[1], sorted_indices[0]  # Ensure Cluster 2 is closest to (0, 0)
            if cluster_2_idx == 0:
                cluster_1_idx = 1 - cluster_1_idx
                cluster_2_idx = 1 - cluster_2_idx
                labels = 1 - labels  # Swap labels
                centroids = centroids[[1, 0]]  # Swap centroids

            # Normalize signal intensities based on Cluster 1
            cluster_1_points = points[labels == cluster_1_idx]
            cluster_1_centroid = centroids[cluster_1_idx]
            cluster_1_std = np.std(cluster_1_points, axis=0)
            centroid_std_dict[slice] = [cluster_1_centroid, cluster_1_std]
        
        return centroid_std_dict
    
    def slider_changed(self, val):
        self.slice = int(val)
        self.redraw_axes()

    def redraw_axes(self):
        self.ax1.clear()
        self.ax2.clear()
        self.img1 = self.ax1.imshow(self.image1_data[:, :, self.slice], cmap='gray', interpolation='none')
        self.ax1.axis('off')
        self.img2 = self.ax2.imshow(self.image2_data[:, :, self.slice], cmap='gray', interpolation='none')
        self.ax2.axis('off')
        self.overlay_mask()

    def on_image_click(self, event):

        if event.inaxes in [self.ax1, self.ax2]:
            if event.inaxes == self.ax1:
                selected_contrast = "t1"
            else:
                selected_contrast = "t2flair"

            self.selected_slice = self.slice
            self.x, self.y = int(event.xdata), int(event.ydata)
        
            if event.button == 1:  # Left mouse button
               
                if self.leftclick == False:
                    self.leftclick = True
                else:
                    self.saved_connected_mask |= self.new_connected_mask
                    self.undo_saved_connected_mask = copy.deepcopy(self.saved_connected_mask) # Creates a copy of the saved_connected_mask to retrieve if user undones

                if self.neighbor_select == False:
                    self.image1_value = self.image1_data[self.y, self.x, self.slice]
                    self.image2_value = self.image2_data[self.y, self.x, self.slice]
                    self.new_connected_mask = self.create_connected()
                    self.new_connected_mask |= self.saved_connected_mask
                    self.filled_mask = self.fill_mask(self.new_connected_mask) 
                    self.overlay_mask()

                else:
                    self.new_connected_mask = self.create_connected_neighbors()
                    self.new_connected_mask |= self.saved_connected_mask
                    self.filled_mask = self.fill_mask(self.new_connected_mask) 
                    self.overlay_mask()  

            elif event.button == 3: # Right mouse button
                #self.saved_connected_mask |= self.new_connected_mask
                if self.filled_mask[self.y, self.x, self.selected_slice] == True:
                    #self.leftclick = False
                    self.save_mask()

                else:
                    print("Selection is not in mask")

    def create_connected(self):
        # Calculate normalized difference array
        image1_diff = np.abs(self.image1_data - self.image1_value) / self.centroid_std_dict[self.selected_slice][1][0] 
        image2_diff = np.abs(self.image2_data - self.image2_value) / self.centroid_std_dict[self.selected_slice][1][1] 
        
        threshold_selection = (image1_diff <= self.adjustable_threshold) & (image2_diff <= self.adjustable_threshold)
        
        # Label connected components
        labeled_mask, num_features = label(threshold_selection)
        selected_label = labeled_mask[self.y, self.x, self.selected_slice]
        new_connected_mask = labeled_mask == selected_label
        return new_connected_mask

    def create_connected_neighbors(self):

        neighbor_connected_mask = np.zeros_like(self.image1_data, dtype=np.uint8).astype(bool)
        neighbors = self.get_neighbors()
        
        for neighbor in neighbors:

            self.image1_value = self.image1_data[neighbor[0], neighbor[1], self.slice]
            self.image2_value = self.image2_data[neighbor[0], neighbor[1], self.slice]

            # Calculate normalized difference array
            image1_diff = np.abs(self.image1_data - self.image1_value) / self.centroid_std_dict[self.selected_slice][1][0] 
            image2_diff = np.abs(self.image2_data - self.image2_value) / self.centroid_std_dict[self.selected_slice][1][1] 
        
            threshold_selection = (image1_diff <= self.adjustable_threshold) & (image2_diff <= self.adjustable_threshold)
        
            # Label connected components
            labeled_mask, num_features = label(threshold_selection)
            selected_label = labeled_mask[neighbor[0], neighbor[1], self.selected_slice]
            new_connected_mask = labeled_mask == selected_label

            neighbor_connected_mask |= new_connected_mask

            # Reset values to original selected point
            self.image1_value = self.image1_data[self.y, self.x, self.slice]
            self.image2_value = self.image2_data[self.y, self.x, self.slice]
        
        return neighbor_connected_mask


    # Function to fill mask
    def fill_mask(self,nii):
        # Dilate mask
        strel = np.ones((3,3,1)) #3D structure element
        dilated_mask = binary_dilation(nii, strel)
        # Fill holes
        filled_mask = binary_fill_holes(dilated_mask)
        return filled_mask

    def overlay_mask(self):
        if self.leftclick:
            mask = self.filled_mask[:, :, self.slice]
            self.mask_overlay = np.zeros((*mask.shape, 4))  # Create an RGBA mask overlay
            self.mask_overlay[mask > 0, 0] = 1  # Red channel for the mask
            self.mask_overlay[mask > 0, 3] = 0.5  # Alpha channel for transparency
            self.ax1.imshow(self.mask_overlay, interpolation='none')
            self.ax2.imshow(self.mask_overlay, interpolation='none')
        self.fig.canvas.draw_idle()

    def get_neighbors(self):
        distance = 1
        neighbors = []
    
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                ny, nx = self.y + dy, self.x + dx
                # Handle boundary conditions by clamping
                nx = np.clip(nx, 0, self.H - 1)
                ny = np.clip(ny, 0, self.W - 1)
                if (ny, nx) not in neighbors:  # Avoid duplicates
                    neighbors.append((ny, nx))
    
        return neighbors


    # Function to handle mouse scroll
    def on_scroll(self, event):
        if self.leftclick == True:
            if event.button == 'down' and self.adjustable_threshold < 1.1:
                self.adjustable_threshold = min(self.adjustable_threshold + 0.05, 1.1)  # Increase threshold
            elif event.button == 'up' and self.adjustable_threshold >0.1:
                self.adjustable_threshold = max(self.adjustable_threshold - 0.05, 0.1)  # Decrease threshold
            print(f"Adjustable threshold updated: {self.adjustable_threshold}")
        
            # Update the mask with new threshold
            if self.neighbor_select == False:
                self.new_connected_mask = self.create_connected()
                self.new_connected_mask |= self.saved_connected_mask
                self.filled_mask = self.fill_mask(self.new_connected_mask)  
            self.redraw_axes()

    # Function to handle key press
    def on_key_press(self, event):
    # Escape key will cancel current mask and reset variables 
        if event.key == 'escape': 
            self.leftclick = False
            self.new_connected_mask = np.zeros_like(self.image1_data, dtype=np.uint8).astype(bool)
            self.saved_connected_mask = np.zeros_like(self.image1_data, dtype=np.uint8).astype(bool)
            self.undo_saved_connected_mask = np.zeros_like(self.image1_data, dtype=np.uint8).astype(bool)
            print("Masks Cleared")  
            self.redraw_axes()

    # 'u' key will undo current mask  
        if event.key == 'u': 
            self.new_connected_mask = np.zeros_like(self.image1_data, dtype=np.uint8).astype(bool)
            self.saved_connected_mask = self.undo_saved_connected_mask
            print("Undo last selection")  
            self.redraw_axes()

    # 'n' key will allow user to select multiple neighbor voxels with click    
        if event.key == 'n':
            self.neighbor_select = not self.neighbor_select 
            if self.neighbor_select:
                print("Nearest neighbor mode on")
            else:
                print("Nearest neighbor mode off")  
                
# User input for path and file names
filepath = "/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom2nifti_upload/DSC_0010/20240422/"
image1_filename = input("Enter name of image 1 dataset as name.nii:") or "t1+_processed.nii.gz"
image2_filename = input("Enter name of image 2 dataset as name.nii:") or "flair+_processed.nii.gz"

# Main Program
if __name__ == "__main__":
    viewer = MRIViewer(filepath, image1_filename, image2_filename)
