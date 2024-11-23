import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import easyocr
import os
import csv
from multiprocessing import Pool, cpu_count
import gc
import json
import logging
import traceback
import shutil
from datetime import datetime

# Configuration class
class Config:
    DEFAULT_PARAMS = {
        'min_density': 0.15,
        'max_density': 0.7,
        'midline_pos': 0.75,
        'min_above_midline': 50.0,
        'min_relative_area': 0.016,
        'ocr_confidence': 0.4
    }
    
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    BATCH_SIZE = 5
    CHUNK_SIZE = 10
    MIN_DISPLAY_WIDTH = 400
    MIN_DISPLAY_HEIGHT = 300

# Initialize global OCR reader
reader = easyocr.Reader(['en'], gpu=False)

class DensityDetectorGUI:
    def __init__(self, root, debug_mode=False):
        self.root = root
        self.root.title("Density Analysis Detector")
        
        # Debug mode
        self.debug_mode = debug_mode
        
        # Cache and state variables
        self.ocr_cache = {}
        self.current_directory = ""
        self.output_directory = os.path.expanduser("~")
        self.image_files = []
        self.current_image_index = -1
        self.current_image = None
        self.ocr_result = ""
        
        # Setup main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create GUI elements
        self.create_widgets()
        
        # Create debug window if in debug mode
        if self.debug_mode:
            self.create_debug_window()
            
        # Bind resize event
        self.canvas.bind('<Configure>', self.on_resize)
        
    def on_resize(self, event=None):
        """Handle resize events"""
        if self.current_image is not None:
            self.root.after(100, self.process_image)  # Debounce resize events
            
    def create_debug_window(self):
        """Create debug window for additional information"""
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Debug Information")
        debug_window.geometry("400x300")
        
        self.debug_text = tk.Text(debug_window, wrap=tk.WORD)
        self.debug_text.pack(fill=tk.BOTH, expand=True)
        
        return debug_window
        
    def debug_print(self, message):
        """Print debug information if in debug mode"""
        if self.debug_mode:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.debug_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.debug_text.see(tk.END)
        print(message)
    def create_widgets(self):
        """Create and arrange all widgets in the GUI."""
        try:
            # Main layout using grid with two columns
            self.main_frame.grid_columnconfigure(1, weight=1)
            
            # Create left and right columns
            self.create_left_column()
            self.create_right_column()
            
            # Configure grid weights to allow expansion
            self.main_frame.grid_rowconfigure(0, weight=1)
            self.main_frame.grid_columnconfigure(1, weight=1)
            
            # Make the window resizable
            self.root.resizable(True, True)
            
        except Exception as e:
            self.show_error("Error creating widgets", e)

    def create_left_column(self):
        """Create left column containing controls and parameters"""
        left_column = ttk.Frame(self.main_frame, padding="5")
        left_column.grid(row=0, column=0, sticky=tk.N+tk.S+tk.W)
        left_column.grid_columnconfigure(0, weight=1)
        
        # Create control panel
        self.create_control_panel(left_column)
        
        # Create parameters panel
        self.create_parameters_panel(left_column)
        
        # Create OCR result panel
        self.create_ocr_panel(left_column)

    def create_control_panel(self, parent):
        """Create control panel with buttons and navigation"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky=tk.E+tk.W, pady=5)
        
        # Load image button
        ttk.Button(control_frame, text="Load Image", 
                command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="Previous", 
                command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", 
                command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Batch process button
        ttk.Button(control_frame, text="Batch Process", 
                command=self.batch_process_images).grid(row=3, column=0, padx=5, pady=5)
        
        # File indicator
        self.file_label = ttk.Label(control_frame, text="No image loaded")
        self.file_label.grid(row=2, column=0, padx=5, pady=5)
        
        # Output directory controls
        self.create_output_dir_controls(control_frame)

    def create_parameters_panel(self, parent):
        """Create parameters panel with sliders and checkboxes"""
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding="5")
        param_frame.grid(row=1, column=0, sticky=tk.E+tk.W, pady=5)
        param_frame.grid_columnconfigure(0, weight=1)
        
        # Create sliders
        self.create_parameter_sliders(param_frame)
        
        # Create checkboxes
        self.create_parameter_checkboxes(param_frame)

    def create_parameter_sliders(self, parent):
        """Create all parameter sliders"""
        param_row = 0
        
        # Min density threshold
        self.min_density_var = tk.DoubleVar(value=Config.DEFAULT_PARAMS['min_density'])
        self.create_slider_with_entry(
            parent, "Min Density", self.min_density_var, 0.1, 0.5, 0.05, param_row
        )
        param_row += 1
        
        # Max density threshold
        self.max_density_var = tk.DoubleVar(value=Config.DEFAULT_PARAMS['max_density'])
        self.create_slider_with_entry(
            parent, "Max Density", self.max_density_var, 0.5, 1, 0.05, param_row
        )
        param_row += 1
        
        # Midline position
        self.midline_pos_var = tk.DoubleVar(value=Config.DEFAULT_PARAMS['midline_pos'])
        self.create_slider_with_entry(
            parent, "Midline Position", self.midline_pos_var, 0.1, 0.9, 0.05, param_row
        )
        param_row += 1
        
        # Min percentage above midline
        self.min_above_midline_var = tk.DoubleVar(value=Config.DEFAULT_PARAMS['min_above_midline'])
        self.create_slider_with_entry(
            parent, "Min % Above Midline", self.min_above_midline_var, 0.0, 100.0, 5.0, param_row
        )
        param_row += 1
        
        # Min relative area
        self.min_relative_area_var = tk.DoubleVar(value=Config.DEFAULT_PARAMS['min_relative_area'])
        self.create_slider_with_entry(
            parent, "Min Relative Area", self.min_relative_area_var, 0.01, 0.09, 0.002, param_row
        )
        param_row += 1
        
        # OCR confidence threshold
        self.ocr_confidence_var = tk.DoubleVar(value=Config.DEFAULT_PARAMS['ocr_confidence'])
        self.create_slider_with_entry(
            parent, "OCR Confidence", self.ocr_confidence_var, 0.0, 1.0, 0.05, param_row
        )

    def create_parameter_checkboxes(self, parent):
        """Create parameter checkboxes"""
        # Allow negative values
        self.allow_negative_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, 
            text="Allow Negative Values",
            variable=self.allow_negative_var,
            command=self.process_image
        ).grid(row=6, column=0, pady=5)
        
        # Show debug info
        self.show_debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent,
            text="Show Debug Info",
            variable=self.show_debug_var,
            command=self.process_image
        ).grid(row=7, column=0, pady=5)

    def create_ocr_panel(self, parent):
        """Create OCR result panel"""
        self.ocr_frame = ttk.LabelFrame(parent, text="OCR Result", padding="5")
        self.ocr_frame.grid(row=2, column=0, sticky=tk.E+tk.W, pady=5)
        
        self.ocr_result_text = tk.Text(self.ocr_frame, height=3, wrap=tk.WORD)
        self.ocr_result_text.grid(row=0, column=0, sticky=tk.E+tk.W, padx=5, pady=5)

    def create_right_column(self):
        """Create right column containing image preview"""
        canvas_frame = ttk.LabelFrame(self.main_frame, text="Image Preview", padding="5")
        canvas_frame.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W, padx=5)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=Config.MIN_DISPLAY_WIDTH * 2,
            height=Config.MIN_DISPLAY_HEIGHT,
            bg='white'
        )
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W, padx=5, pady=5)

    def show_error(self, title, error):
        """Show error in a popup window"""
        error_msg = f"{title}: {str(error)}\n{traceback.format_exc()}"
        self.debug_print(error_msg)
        
        error_window = tk.Toplevel(self.root)
        error_window.title("Error")
        error_window.geometry("600x400")
        
        error_text = tk.Text(error_window)
        error_text.pack(fill=tk.BOTH, expand=True)
        error_text.insert(tk.END, error_msg)
        
        ttk.Button(
            error_window,
            text="Close",
            command=error_window.destroy
        ).pack()
        
        error_window.lift()
        error_window.focus_force()
        
    def process_image(self):
        """Process the current image with all configured parameters"""
        if self.current_image is None:
            self.debug_print("No image loaded")
            return
            
        try:
            self.debug_print("Processing image...")
            self.root.config(cursor="watch")
            self.root.update()
            
            # Convert to grayscale
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            # Process components
            result_image, valid_components, minus_signs = self.process_components(gray, self.current_image)
            
            # Perform OCR
            ocr_result, processed_regions = self.perform_ocr(self.current_image, valid_components, minus_signs)
            
            # Update OCR result display
            self.ocr_result_text.delete('1.0', tk.END)
            self.ocr_result_text.insert('1.0', f"Detected Number: {ocr_result}")
            
            # Draw OCR results on debug image
            if self.show_debug_var.get():
                for region in processed_regions:
                    x1, y1, x2, y2 = region['region']
                    conf = region['confidence']
                    text = region['text']
                    if conf >= self.ocr_confidence_var.get():
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        cv2.putText(
                            result_image,
                            f"{text} ({conf:.2f})",
                            (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1
                        )
            
            # Display the results
            if self.current_image is not None and result_image is not None:
                self.debug_print("Displaying images...")
                # Create copies to avoid modifying originals
                original_copy = self.current_image.copy()
                result_copy = result_image.copy()
                self.display_images(original_copy, result_copy)
            else:
                self.debug_print("Invalid images for display")
                
        except Exception as e:
            self.show_error("Error in process_image", e)
        finally:
            self.root.config(cursor="")

    def display_images(self, original, processed):
        """Display original and processed images side by side"""
        try:
            self.debug_print("Display images called")
            self.debug_print(f"Original shape: {original.shape if original is not None else None}")
            self.debug_print(f"Processed shape: {processed.shape if processed is not None else None}")
            
            # Validate images
            if original is None or processed is None:
                raise ValueError("Invalid image data")
                
            if original.shape[0] == 0 or original.shape[1] == 0:
                raise ValueError("Invalid image dimensions")
            
            # Get canvas dimensions
            canvas_width = max(self.canvas.winfo_width(), Config.MIN_DISPLAY_WIDTH * 2)
            canvas_height = max(self.canvas.winfo_height(), Config.MIN_DISPLAY_HEIGHT)
            
            self.debug_print(f"Canvas dimensions: {canvas_width}x{canvas_height}")
            
            # Calculate available space
            available_width = (canvas_width - 20) // 2  # 20 pixels spacing between images
            available_height = canvas_height
            
            # Calculate scale to fit both images
            height, width = original.shape[:2]
            scale = min(available_height/height, available_width/width)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            self.debug_print(f"New dimensions: {new_width}x{new_height}")
            
            # Resize and convert images
            original_resized = cv2.resize(original, (new_width, new_height))
            processed_resized = cv2.resize(processed, (new_width, new_height))
            
            # Convert to PIL format
            original_pil = Image.fromarray(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
            processed_pil = Image.fromarray(cv2.cvtColor(processed_resized, cv2.COLOR_BGR2RGB))
            
            # Convert to PhotoImage
            self.original_tk = ImageTk.PhotoImage(original_pil)
            self.processed_tk = ImageTk.PhotoImage(processed_pil)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.original_tk)
            self.canvas.create_image(new_width + 20, 0, anchor=tk.NW, image=self.processed_tk)
            
        except Exception as e:
            self.show_error("Error in display_images", e)

    def calculate_box_above_midline(self, box_coords, img_height):
        """Calculate percentage of box above midline"""
        x, y, w, h = box_coords
        y_upper = y
        y_lower = y + h
        
        # Calculate midline based on user-defined position
        midline = int(img_height * self.midline_pos_var.get())
        
        # If box is completely above midline
        if y_lower <= midline:
            return 100.0
        
        # If box is completely below midline
        if y_upper >= midline:
            return 0.0
        
        # Box crosses the midline - calculate percentage above
        percentage = ((midline - y_upper) / h) * 100
        return percentage

    def calculate_box_above_upperline(self, box_coords, img_height):
        """Calculate percentage of box above upper line"""
        x, y, w, h = box_coords
        y_upper = y
        y_lower = y + h
        
        # Fixed upper line at 15% of image height
        upperline = int(img_height * 0.15)
        
        # If box is completely above upperline
        if y_lower <= upperline:
            return 100.0
        
        # If box is completely below upperline
        if y_upper >= upperline:
            return 0.0
        
        # Box crosses the upperline - calculate percentage above
        percentage = ((upperline - y_upper) / h) * 100
        return percentage

    def draw_debug_info(self, image, x, y, density, mid_percent, upper_percent, aspect_ratio, color):
        """Draw debug information on the image"""
        info_lines = [
            f'D: {density:.2f}',
            f'M: {mid_percent:.1f}%',
            f'U: {upper_percent:.1f}%',
            f'AR: {aspect_ratio:.2f}'
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = y - 5 - (15 * i)  # 15 pixels spacing between lines
            cv2.putText(
                image,
                line,
                (x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    def process_components(self, gray, image):
        """Process image components and detect numbers and minus signs"""
        try:
            # Threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Get components with stats
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Create debug image
            debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Draw reference lines
            midline_y = int(gray.shape[0] * self.midline_pos_var.get())
            upperline_y = int(gray.shape[0] * 0.15)
            cv2.line(debug_image, (0, midline_y), (gray.shape[1], midline_y), (255, 0, 0), 1)  # Blue midline
            cv2.line(debug_image, (0, upperline_y), (gray.shape[1], upperline_y), (255, 255, 0), 1)  # Yellow upperline

            # Calculate metrics using numpy operations
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            total_image_area = gray.shape[0] * gray.shape[1]
            relative_areas = areas / total_image_area
            
            widths = stats[1:, cv2.CC_STAT_WIDTH]
            heights = stats[1:, cv2.CC_STAT_HEIGHT]
            aspect_ratios = np.divide(widths, heights, where=heights!=0)
            
            # Calculate densities
            densities = np.array([
                np.sum(binary[
                    stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT],
                    stats[i, cv2.CC_STAT_LEFT]:stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                ]) / (255.0 * stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT])
                for i in range(1, num_labels)
            ])

            valid_components = []
            minus_signs = []
            
            # Process each component
            for i in range(1, num_labels):
                idx = i - 1  # Adjust index for arrays that skip background
                
                # Extract component properties
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate position relative to lines
                box_coords = (x, y, w, h)
                percent_above_mid = self.calculate_box_above_midline(box_coords, gray.shape[0])
                percent_above_upper = self.calculate_box_above_upperline(box_coords, gray.shape[0])
                
                # Draw initial rectangle in red
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                
                # Check if component is a valid number
                is_valid_number = (
                    self.min_density_var.get() < densities[idx] < self.max_density_var.get() and 
                    percent_above_mid >= self.min_above_midline_var.get() and
                    percent_above_upper < 60 and
                    0.3 <= aspect_ratios[idx] <= 6 and
                    relative_areas[idx] >= self.min_relative_area_var.get() and
                    aspect_ratios[idx] < 0.8 and 
                    x > 5
                )
                
                if is_valid_number:
                    # Store valid component info
                    valid_components.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'density': densities[idx],
                        'mid_percent': percent_above_mid,
                        'upper_percent': percent_above_upper,
                        'aspect_ratio': aspect_ratios[idx],
                        'centroid': centroids[i]
                    })
                    
                    # Draw valid number in green
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add debug information
                    if self.show_debug_var.get():
                        self.draw_debug_info(
                            debug_image, x, y,
                            densities[idx], percent_above_mid,
                            percent_above_upper, aspect_ratios[idx],
                            (0, 255, 0)
                        )
                else:
                    # Check if component is a minus sign
                    is_minus = (
                        self.allow_negative_var.get() and
                        ((1.8 <= aspect_ratios[idx] <= 3 and densities[idx] > 0.8) or 
                        (1.3 <= aspect_ratios[idx] < 1.8 and densities[idx] > 0.9)) and
                        relative_areas[idx] >= 0.001 and
                        percent_above_mid >= self.min_above_midline_var.get() and
                        percent_above_upper < 60
                    )
                    
                    if is_minus:
                        # Store minus sign info
                        minus_signs.append({
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'density': densities[idx],
                            'aspect_ratio': aspect_ratios[idx],
                            'relative_area': relative_areas[idx],
                            'centroid': centroids[i]
                        })
                        
                        # Draw minus sign in blue
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        
                        # Add debug information for minus signs
                        if self.show_debug_var.get():
                            self.draw_debug_info(
                                debug_image, x, y,
                                densities[idx], percent_above_mid,
                                percent_above_upper, aspect_ratios[idx],
                                (255, 0, 0)
                            )
            
            return debug_image, valid_components, minus_signs
            
        except Exception as e:
            self.show_error("Error in process_components", e)
            return None, [], []

    def perform_ocr(self, image, components, minus_signs):
        """Perform OCR on detected components"""
        try:
            global reader
            
            # Generate cache key
            cache_key = hash(str(components) + str(minus_signs) + str(self.ocr_confidence_var.get()))
            if cache_key in self.ocr_cache:
                return self.ocr_cache[cache_key]

            result = ""
            processed_regions = []
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            min_confidence = self.ocr_confidence_var.get()

            # Process components in batches
            batch_size = Config.BATCH_SIZE
            for i in range(0, len(components), batch_size):
                batch = components[i:i + batch_size]
                regions = []
                region_coords = []

                # Prepare regions for OCR
                for comp in batch:
                    x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
                    pad = int(h * 0.1)  # Add 10% padding
                    y1 = max(0, y - pad)
                    y2 = min(image.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(image.shape[1], x + w + pad)
                    
                    region = rgb_image[y1:y2, x1:x2]
                    regions.append(region)
                    region_coords.append((x1, y1, x2, y2, comp))

                # Process each region
                for region, (x1, y1, x2, y2, comp) in zip(regions, region_coords):
                    ocr_result = reader.readtext(
                        region,
                        allowlist='0123456789',
                        min_size=10,
                        paragraph=False
                    )
                    
                    digit_found = False
                    for box, text, confidence in ocr_result:
                        if confidence >= min_confidence:
                            digit = text[0] if len(text) > 0 else text
                            result += digit
                            processed_regions.append({
                                'region': (x1, y1, x2, y2),
                                'text': text,
                                'confidence': confidence
                            })
                            digit_found = True
                            break
                    
                    # Apply rule-based detection if OCR fails
                    if not digit_found:
                        if comp['aspect_ratio'] < 0.5 and comp['density'] > 0.3:
                            result += "1"
                            processed_regions.append({
                                'region': (x1, y1, x2, y2),
                                'text': "1",
                                'confidence': 1.0
                            })
                        elif comp['aspect_ratio'] > 0.5 and comp['density'] < 0.4:
                            result += "7"
                            processed_regions.append({
                                'region': (x1, y1, x2, y2),
                                'text': "7",
                                'confidence': 1.0
                            })

            # Add minus sign if detected
            if minus_signs:
                result = f"-{result}"
            
            # Cache and return results
            self.ocr_cache[cache_key] = (result, processed_regions)
            return result, processed_regions
            
        except Exception as e:
            self.show_error("Error in perform_ocr", e)
            return "", []
    def load_image(self):
        """Load a single image and initialize batch processing if needed"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", " ".join(f"*{ext}" for ext in Config.IMAGE_EXTENSIONS))]
            )
            if file_path:
                # Store directory and get all image files
                self.current_directory = os.path.dirname(file_path)
                self.image_files = self.get_image_files()
                
                # Find index of selected file
                current_file = os.path.basename(file_path)
                self.current_image_index = self.image_files.index(current_file)
                
                # Load the image
                self.load_image_at_index(self.current_image_index)
                
        except Exception as e:
            self.show_error("Error loading image", e)

    def get_image_files(self):
        """Get all valid image files from current directory"""
        try:
            files = [f for f in os.listdir(self.current_directory) 
                    if f.lower().endswith(Config.IMAGE_EXTENSIONS)]
            return sorted(files)
        except Exception as e:
            self.show_error("Error getting image files", e)
            return []

    def load_image_at_index(self, index):
        """Load image at specified index"""
        if 0 <= index < len(self.image_files):
            try:
                file_path = os.path.join(self.current_directory, self.image_files[index])
                self.debug_print(f"Loading image: {file_path}")
                
                # Validate file
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                if os.path.getsize(file_path) == 0:
                    raise ValueError(f"Empty file: {file_path}")
                
                # Load image
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError(f"Failed to load image: {file_path}")
                
                self.debug_print(f"Image loaded successfully. Shape: {self.current_image.shape}")
                self.current_image_index = index
                
                # Update UI
                self.file_label.config(
                    text=f"Image {index + 1} of {len(self.image_files)}: {self.image_files[index]}"
                )
                
                # Process image
                self.process_image()
                
            except Exception as e:
                self.show_error("Error loading image at index", e)

    def batch_process_images(self):
        """Process all images in the current directory"""
        if not self.current_directory or not self.image_files:
            messagebox.showerror("Error", "Please load an image first.")
            return

        if not messagebox.askyesno("Confirm", "Start batch processing?"):
            return

        try:
            # Prepare progress tracking
            total_files = len(self.image_files)
            progress_window = self.create_progress_window(total_files)
            progress_var = progress_window.progress_var
            
            # Prepare parameters
            params = self.get_current_parameters()
            
            # Process in chunks
            results = []
            for chunk_start in range(0, total_files, Config.CHUNK_SIZE):
                chunk_end = min(chunk_start + Config.CHUNK_SIZE, total_files)
                chunk = self.image_files[chunk_start:chunk_end]
                
                # Process chunk
                chunk_results = self.process_image_chunk(chunk, params, progress_var)
                results.extend(chunk_results)
                
                # Update progress
                progress_var.set(chunk_end)
                progress_window.update()
                
                # Clear memory
                gc.collect()
                self.ocr_cache.clear()
            
            # Save results
            if results:
                self.save_results_to_csv(results)
                messagebox.showinfo(
                    "Success",
                    f"Processed {len(results)} images successfully."
                )
            else:
                messagebox.showwarning("Warning", "No valid results were generated.")
            
            progress_window.destroy()
            
        except Exception as e:
            self.show_error("Batch processing failed", e)

    def create_progress_window(self, total_files):
        """Create progress window for batch processing"""
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Images")
        progress_window.geometry("300x150")
        progress_window.transient(self.root)
        
        # Progress bar
        progress_window.progress_var = tk.IntVar()
        ttk.Label(
            progress_window,
            text="Processing images..."
        ).pack(pady=10)
        
        ttk.Progressbar(
            progress_window,
            variable=progress_window.progress_var,
            maximum=total_files,
            length=200,
            mode='determinate'
        ).pack(pady=10)
        
        # Progress label
        progress_window.label = ttk.Label(
            progress_window,
            text="0/{total_files} files processed"
        )
        progress_window.label.pack(pady=10)
        
        return progress_window

    def process_image_chunk(self, chunk, params, progress_var=None):
        """Process a chunk of images in parallel"""
        try:
            # Prepare arguments for parallel processing
            image_paths = [os.path.join(self.current_directory, img) for img in chunk]
            args = [(path, params) for path in image_paths]
            
            # Process in parallel
            with Pool(processes=max(1, cpu_count() - 1)) as pool:
                results = pool.map(self.process_single_image_parallel, args)
            
            # Filter and validate results
            valid_results = []
            for result in results:
                if result and isinstance(result, dict):
                    if 'time' in result and 'value' in result:
                        valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            self.show_error("Error processing image chunk", e)
            return []

    def process_single_image_parallel(self, args):
        """Process a single image in parallel"""
        img_path, params = args
        try:
            # Early exit conditions
            if not os.path.exists(img_path):
                return None
                
            image = cv2.imread(img_path)
            if image is None or image.shape[0] < 10 or image.shape[1] < 10:
                return None

            # Extract time from filename
            try:
                time_value = int(''.join(filter(str.isdigit, os.path.basename(img_path))))
            except ValueError:
                return None

            # Process image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            debug_image, valid_components, minus_signs = self.process_components(gray, image)
            ocr_result, _ = self.perform_ocr(image, valid_components, minus_signs)

            # Convert result to number
            try:
                numeric_value = float(ocr_result)
                return {
                    'time': time_value,
                    'value': numeric_value,
                    'path': img_path
                }
            except ValueError:
                return {
                    'time': time_value,
                    'value': float('nan'),
                    'path': img_path
                }

        except Exception as e:
            self.debug_print(f"Error processing {img_path}: {str(e)}")
            return None

    def save_results_to_csv(self, results):
        """Save processing results to CSV file"""
        try:
            dir_name = os.path.basename(self.current_directory)
            output_path = os.path.join(self.output_directory, f"{dir_name}.csv")
            
            # Create backup if file exists
            if os.path.exists(output_path):
                backup_path = f"{output_path}.bak"
                shutil.copy2(output_path, backup_path)
            
            # Sort results by time
            results.sort(key=lambda x: x['time'])
            
            # Save to CSV
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Time', 'Value', 'File'])  # Header
                
                for result in results:
                    value = result.get('value', float('nan'))
                    if not np.isnan(value):
                        value = int(value)
                    writer.writerow([
                        result['time'],
                        value,
                        os.path.basename(result.get('path', ''))
                    ])
                    
            self.debug_print(f"Results saved to: {output_path}")
            
        except Exception as e:
            self.show_error("Error saving results", e)

    def get_current_parameters(self):
        """Get current parameter values"""
        return {
            'min_density': self.min_density_var.get(),
            'max_density': self.max_density_var.get(),
            'midline_pos': self.midline_pos_var.get(),
            'min_above_midline': self.min_above_midline_var.get(),
            'min_relative_area': self.min_relative_area_var.get(),
            'ocr_confidence': self.ocr_confidence_var.get(),
            'allow_negative': self.allow_negative_var.get()
        }
    def next_image(self):
        """Load the next image in the directory"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.load_image_at_index(self.current_image_index + 1)

    def prev_image(self):
        """Load the previous image in the directory"""
        if self.image_files and self.current_image_index > 0:
            self.load_image_at_index(self.current_image_index - 1)

    def select_output_directory(self):
        """Allow user to select output directory for CSV files"""
        try:
            directory = filedialog.askdirectory(
                initialdir=self.output_directory,
                title="Select Output Directory for CSV Files"
            )
            if directory:
                self.output_directory = directory
                self.output_dir_label.config(text=f"Output Dir: {self.output_directory}")
                self.debug_print(f"Output directory set to: {self.output_directory}")
        except Exception as e:
            self.show_error("Error selecting output directory", e)

    def create_output_dir_controls(self, control_frame):
        """Create output directory selection controls"""
        try:
            output_dir_frame = ttk.Frame(control_frame)
            output_dir_frame.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
            
            ttk.Button(
                output_dir_frame, 
                text="Set Output Directory", 
                command=self.select_output_directory
            ).pack(side=tk.LEFT, padx=5)
            
            self.output_dir_label = ttk.Label(
                output_dir_frame, 
                text=f"Output Dir: {self.output_directory}"
            )
            self.output_dir_label.pack(side=tk.LEFT, padx=5)
        except Exception as e:
            self.show_error("Error creating output directory controls", e)

    def validate_slider_values(self, param_name, value, min_val, max_val, step):
        """Validate slider input values against constraints"""
        try:
            value = float(value)
            if not (min_val <= value <= max_val):
                messagebox.showwarning(
                    "Invalid Value",
                    f"{param_name} must be between {min_val} and {max_val}"
                )
                return False
                
            # Check if value aligns with step size
            steps = round((value - min_val) / step)
            valid_value = min_val + (steps * step)
            
            # Use small epsilon for float comparison
            if abs(valid_value - value) > 1e-10:
                messagebox.showwarning(
                    "Invalid Step",
                    f"{param_name} must be in steps of {step}"
                )
                return False
            return True
            
        except ValueError:
            messagebox.showerror("Error", f"{param_name} must be a number")
            return False

    def reset_to_defaults(self):
        """Reset all parameters to default values"""
        try:
            self.min_density_var.set(Config.DEFAULT_PARAMS['min_density'])
            self.max_density_var.set(Config.DEFAULT_PARAMS['max_density'])
            self.midline_pos_var.set(Config.DEFAULT_PARAMS['midline_pos'])
            self.min_above_midline_var.set(Config.DEFAULT_PARAMS['min_above_midline'])
            self.min_relative_area_var.set(Config.DEFAULT_PARAMS['min_relative_area'])
            self.ocr_confidence_var.set(Config.DEFAULT_PARAMS['ocr_confidence'])
            self.allow_negative_var.set(True)
            self.show_debug_var.set(True)
            
            if self.current_image is not None:
                self.process_image()
                
            self.debug_print("Parameters reset to defaults")
        except Exception as e:
            self.show_error("Error resetting parameters", e)

    def save_configuration(self):
        """Save current configuration to file"""
        try:
            config_path = os.path.join(self.output_directory, "detector_config.json")
            config = {
                'parameters': self.get_current_parameters(),
                'output_directory': self.output_directory
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            self.debug_print(f"Configuration saved to: {config_path}")
        except Exception as e:
            self.show_error("Error saving configuration", e)

    def load_configuration(self):
        """Load configuration from file"""
        try:
            config_path = os.path.join(self.output_directory, "detector_config.json")
            if not os.path.exists(config_path):
                return
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load parameters
            params = config.get('parameters', {})
            for param, value in params.items():
                if hasattr(self, f"{param}_var"):
                    getattr(self, f"{param}_var").set(value)
                    
            # Load output directory
            output_dir = config.get('output_directory')
            if output_dir and os.path.exists(output_dir):
                self.output_directory = output_dir
                self.output_dir_label.config(text=f"Output Dir: {self.output_directory}")
                
            self.debug_print("Configuration loaded")
        except Exception as e:
            self.show_error("Error loading configuration", e)

    def create_menu(self):
        """Create application menu"""
        try:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Load Image", command=self.load_image)
            file_menu.add_command(label="Set Output Directory", command=self.select_output_directory)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.root.quit)
            
            # Edit menu
            edit_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Edit", menu=edit_menu)
            edit_menu.add_command(label="Reset to Defaults", command=self.reset_to_defaults)
            edit_menu.add_separator()
            edit_menu.add_command(label="Save Configuration", command=self.save_configuration)
            edit_menu.add_command(label="Load Configuration", command=self.load_configuration)
            
            # View menu
            view_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="View", menu=view_menu)
            view_menu.add_checkbutton(label="Show Debug Info", variable=self.show_debug_var,
                                    command=self.process_image)
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="About", command=self.show_about)
        except Exception as e:
            self.show_error("Error creating menu", e)

    def show_about(self):
        """Show about dialog"""
        about_text = """
        Density Analysis Detector
        Version 2.0
        
        A tool for analyzing and detecting numbers in images.
        
        Features:
        - Image processing and analysis
        - OCR number detection
        - Batch processing
        - Configuration management
        """
        
        messagebox.showinfo("About", about_text)

# Main execution
if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('detector.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create and run application
        root = tk.Tk()
        root.title("Density Analysis Detector")
        
        # Set minimum window size
        root.minsize(1024, 768)
        
        # Create application instance
        app = DensityDetectorGUI(root, debug_mode=True)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start application
        root.mainloop()
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        
        # Show error in dialog
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Critical Error", error_msg)
        
        # Wait for user input before closing
        input("Press Enter to close...")