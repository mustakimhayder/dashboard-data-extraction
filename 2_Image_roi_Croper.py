import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import signal
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

class ROISelector:
    def __init__(self):
        self.rois = []
        self.image = None
        self.original_image = None
        self.scale_factor = 1.0
        
    def resize_image_to_fit_screen(self, image):
        """Resize image to fit screen while maintaining aspect ratio"""
        root = tk.Tk()
        screen_height = root.winfo_screenheight() - 100
        screen_width = root.winfo_screenwidth() - 100
        root.destroy()
        
        height, width = image.shape[:2]
        scale_w = screen_width / width
        scale_h = screen_height / height
        self.scale_factor = min(scale_w, scale_h)
        
        if self.scale_factor < 1:
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            return cv2.resize(image, (new_width, new_height))
        return image.copy()
    
    def select_rois(self, image_path):
        """Opens a window to select multiple ROIs and returns their coordinates"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise Exception("Could not load image")
            
        self.image = self.resize_image_to_fit_screen(self.original_image)
        
        window_name = "Select ROIs (Draw rectangles with mouse, 'A' to add ROI, SPACE when done, ESC to cancel)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_roi = None
        
        try:
            while True:
                display_image = self.image.copy()
                
                for idx, roi in enumerate(self.rois):
                    x, y, w, h = roi
                    cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_image, str(idx+1), (x+5, y+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.drawing and self.start_point and self.end_point:
                    cv2.rectangle(display_image, self.start_point, self.end_point, (255, 0, 0), 2)
                
                cv2.imshow(window_name, display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('a') and self.current_roi is not None:
                    self.rois.append(self.current_roi)
                    self.current_roi = None
                    print(f"ROI {len(self.rois)} added")
                
                if key == 32 and len(self.rois) > 0:
                    cv2.destroyAllWindows()
                    if self.scale_factor < 1:
                        scaled_rois = []
                        for roi in self.rois:
                            x, y, w, h = roi
                            scaled_rois.append((
                                int(x / self.scale_factor),
                                int(y / self.scale_factor),
                                int(w / self.scale_factor),
                                int(h / self.scale_factor)
                            ))
                        return scaled_rois
                    return self.rois
                
                if key == 27:
                    cv2.destroyAllWindows()
                    return None
                
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            self.current_roi = (x, y, w, h)

class BatchProcessorGUI:
    def __init__(self, rois):
        self.rois = rois
        self.root = tk.Tk()
        self.root.title("ROI Batch Processor")
        self.setup_gui()
        
    def setup_gui(self):
        # Configure main window
        self.root.geometry("600x400")
        self.root.minsize(500, 300)
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # ROI Information
        roi_frame = ttk.LabelFrame(main_frame, text="ROI Information", padding="5")
        roi_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(roi_frame, text=f"Number of ROIs selected: {len(self.rois)}").pack()
        
        # Directory Selection
        dir_frame = ttk.LabelFrame(main_frame, text="Input Directory", padding="5")
        dir_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.dir_var = tk.StringVar()
        dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=50)
        dir_entry.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        browse_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_directory)
        browse_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Process Button and Progress
        self.process_btn = ttk.Button(main_frame, text="Batch Process", command=self.start_processing)
        self.process_btn.grid(row=2, column=0, columnspan=2, pady=20)
        
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Status Text
        self.status_text = tk.Text(main_frame, height=10, width=50)
        self.status_text.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Add scrollbar to status text
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=4, column=2, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        dir_frame.columnconfigure(0, weight=1)
        
    def browse_directory(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.dir_var.set(directory)
            
    def log_message(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def start_processing(self):
        input_dir = self.dir_var.get()
        if not input_dir:
            messagebox.showerror("Error", "Please select an input directory")
            return
            
        self.process_btn.configure(state="disabled")
        self.progress_var.set("Processing...")
        self.log_message(f"\nStarting batch processing for directory: {input_dir}")
        
        try:
            processed_count, error_count, output_location = crop_images_parallel(
                input_dir, 
                self.rois,
                status_callback=self.log_message
            )
            
            if processed_count > 0:
                message = (
                    f"Successfully processed {processed_count} images with {len(self.rois)} ROIs each!\n"
                    + (f"Errors encountered: {error_count}\n" if error_count > 0 else "")
                    + f"\nCropped images are saved in:\n{output_location}"
                )
                messagebox.showinfo("Processing Complete", message)
                self.log_message("\n" + message)
            else:
                messagebox.showerror(
                    "Processing Failed",
                    "No images were processed successfully.\n"
                    "Please check if the input directory contains valid image files (.jpg, .png, .jpeg)"
                )
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.log_message(f"Error: {str(e)}")
        
        self.progress_var.set("")
        self.process_btn.configure(state="normal")
    
    def run(self):
        self.root.mainloop()

def process_image(filename, input_dir, output_dirs, rois):
    """Process a single image with all ROIs"""
    try:
        input_path = os.path.join(input_dir, filename)
        img = cv2.imread(input_path)
        
        for (x, y, w, h), roi_dir in zip(rois, output_dirs):
            cropped = img[y:y+h, x:x+w]
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(roi_dir, f'{base_name}{ext}')
            cv2.imwrite(output_path, cropped)
        
        return True, None
    except Exception as e:
        return False, str(e)

def crop_images_parallel(input_dir, rois, status_callback=None):
    """Crop all images in input_dir using the specified ROIs in parallel"""
    # Get the folder name from the input directory
    folder_name = os.path.basename(input_dir)
    
    # Create output directories within input directory
    roi_dirs = []
    for idx in range(len(rois)):
        roi_dir = os.path.join(input_dir, f'{folder_name}_roi_{idx+1}')
        os.makedirs(roi_dir, exist_ok=True)
        roi_dirs.append(roi_dir)
        if status_callback:
            status_callback(f"Created output directory: {roi_dir}")
    
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        return 0, 0, None
    
    # Use all available CPU cores except one
    num_processes = max(1, cpu_count() - 1)
    if status_callback:
        status_callback(f"\nProcessing {len(image_files)} images using {num_processes} processes...")
    
    process_func = partial(process_image, 
                         input_dir=input_dir,
                         output_dirs=roi_dirs,
                         rois=rois)
    
    processed_count = 0
    error_count = 0
    
    with Pool(num_processes) as pool:
        for i, (success, error) in enumerate(pool.imap_unordered(process_func, image_files), 1):
            if success:
                processed_count += 1
            else:
                error_count += 1
            if status_callback and i % 10 == 0:  # Update status every 10 images
                status_callback(f"Processed {i}/{len(image_files)} images...")
    
    return processed_count, error_count, roi_dirs[0].rsplit('/', 1)[0]

def save_rois(rois, filepath):
    """Save ROI coordinates to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(rois, f)

def main():
    signal.signal(signal.SIGINT, lambda signum, frame: handle_interrupt())
    
    print("Please select a sample image for ROI selection...")
    sample_image = filedialog.askopenfilename(
        title="Select Sample Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
    )
    
    if not sample_image:
        print("No image selected. Exiting...")
        return
    
    roi_selector = ROISelector()
    rois = roi_selector.select_rois(sample_image)
    
    if not rois:
        print("ROI selection cancelled. Exiting...")
        return
    
    save_rois(rois, "roi_coordinates.json")
    print(f"ROIs saved: {rois}")
    
    # Create and run the batch processor GUI
    batch_processor = BatchProcessorGUI(rois)
    batch_processor.run()

def handle_interrupt():
    """Handle Ctrl+C interrupt"""
    print("\nInterrupted by user. Cleaning up...")
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    main()