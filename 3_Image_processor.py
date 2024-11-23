import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tkinter import messagebox

class FilterState:
    def __init__(self):
        self.enabled = tk.BooleanVar(value=False)
        self.params = {}
    
    def add_param(self, name, default_value, min_val, max_val, resolution=1):
        self.params[name] = {
            'value': tk.DoubleVar(value=default_value),
            'min': min_val,
            'max': max_val,
            'resolution': resolution
        }

class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processor")
        
        # Image processing variables
        self.current_directory = ""
        self.output_base_directory = ""
        self.image_files = []
        self.current_image_index = -1
        self.original_image = None
        self.processed_image = None
        
        # Initialize filters
        self.init_filters()
        
        # Setup GUI
        self.setup_gui()
        
        # Bind events
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
    def init_filters(self):
        self.filters = {}
        
        # Gaussian Blur
        self.filters['gaussian'] = FilterState()
        self.filters['gaussian'].add_param('kernel_size', 3, 1, 31, 2)
        self.filters['gaussian'].add_param('sigma', 0, 0, 10, 0.1)
        
        # Bilateral Filter
        self.filters['bilateral'] = FilterState()
        self.filters['bilateral'].add_param('d', 9, 5, 25, 2)
        self.filters['bilateral'].add_param('sigma_color', 75, 10, 150, 1)
        self.filters['bilateral'].add_param('sigma_space', 75, 10, 150, 1)
        
        # Brightness/Contrast
        self.filters['brightness_contrast'] = FilterState()
        self.filters['brightness_contrast'].add_param('brightness', 0, -50, 50, 1)
        self.filters['brightness_contrast'].add_param('contrast', 1, 0.1, 3.0, 0.1)
        
        # CLAHE
        self.filters['clahe'] = FilterState()
        self.filters['clahe'].add_param('clip_limit', 2.0, 0.1, 5.0, 0.1)
        self.filters['clahe'].add_param('grid_size', 8, 2, 16, 1)
        
        # Threshold
        self.filters['threshold'] = FilterState()
        self.filters['threshold'].add_param('block_size', 11, 3, 99, 2)
        self.filters['threshold'].add_param('C', 2, -20, 20, 1)
        
        # Morphological Operations
        self.filters['morphology'] = FilterState()
        self.filters['morphology'].add_param('kernel_size', 3, 1, 15, 2)
        
        # Canny Edge
        self.filters['canny'] = FilterState()
        self.filters['canny'].add_param('threshold1', 100, 0, 255, 1)
        self.filters['canny'].add_param('threshold2', 200, 0, 255, 1)
        
        # Gamma Correction
        self.filters['gamma'] = FilterState()
        self.filters['gamma'].add_param('gamma', 1.0, 0.1, 5.0, 0.1)
        
        # Sharpen
        self.filters['sharpen'] = FilterState()
        self.filters['sharpen'].add_param('amount', 1.0, 0.1, 5.0, 0.1)
        
        # Denoise
        self.filters['denoise'] = FilterState()
        self.filters['denoise'].add_param('h', 10, 0, 50, 1)
    def setup_gui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (controls)
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Right panel (images)
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create controls in left panel
        self.create_controls(left_panel)
        
        # Create image displays in right panel
        self.create_image_displays(right_panel)

    def select_output_directory(self):
        output_dir = filedialog.askdirectory(title="Select Output Base Directory")
        if output_dir:
            self.output_base_directory = output_dir
            # Update label to show selected directory
            self.output_dir_label.config(text=f"Output Directory: {output_dir}")
        
    def create_controls(self, parent):
        # Directory controls frame
        dir_frame = ttk.Frame(parent)
        dir_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Input directory selection
        ttk.Button(dir_frame, text="Select Input Directory", 
                command=self.select_directory).pack(fill=tk.X, pady=(0, 5))
        
        # Output directory selection
        ttk.Button(dir_frame, text="Select Output Directory", 
                command=self.select_output_directory).pack(fill=tk.X, pady=(0, 5))
        
        # Output directory label
        self.output_dir_label = ttk.Label(dir_frame, text="Output Directory: Not Selected", 
                                        wraplength=200)
        self.output_dir_label.pack(fill=tk.X, pady=(0, 5))
    
        
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, expand=True, padx=2)
        
        batch_frame = ttk.Frame(parent)
        batch_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(batch_frame, text="Batch Process Directory", 
                command=self.batch_process_directory).pack(fill=tk.X)
        # Create scrollable frame for filters
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create filter controls
        self.create_filter_controls(scrollable_frame)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_filter_controls(self, parent):
        for filter_name, filter_state in self.filters.items():
            frame = ttk.LabelFrame(parent, text=filter_name.replace('_', ' ').title())
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Enable/disable checkbox
            enable_var = filter_state.enabled
            ttk.Checkbutton(frame, text="Enable", variable=enable_var,
                          command=lambda f=filter_name: self.toggle_filter(f)).pack(anchor=tk.W)
            
            # Parameter sliders
            sliders_frame = ttk.Frame(frame)
            sliders_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Store reference to sliders frame
            filter_state.slider_frame = sliders_frame
            
            # Create sliders for parameters
            for param_name, param_info in filter_state.params.items():
                self.create_parameter_slider(sliders_frame, param_name, param_info)
            
            # Initially hide sliders if filter is disabled
            if not enable_var.get():
                sliders_frame.pack_forget()

    def create_parameter_slider(self, parent, param_name, param_info):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        # Label with value display
        label_frame = ttk.Frame(frame)
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text=param_name.replace('_', ' ').title()).pack(side=tk.LEFT)
        value_label = ttk.Label(label_frame, width=8)
        value_label.pack(side=tk.RIGHT)
        
        # Slider
        slider = ttk.Scale(frame, from_=param_info['min'], to=param_info['max'],
                          variable=param_info['value'], orient=tk.HORIZONTAL)
        slider.pack(fill=tk.X)
        
        # Update value label when slider changes
        def update_label(*args):
            value = param_info['value'].get()
            if param_info['resolution'] >= 1:
                value_label.config(text=f"{int(value)}")
            else:
                value_label.config(text=f"{value:.1f}")
            self.update_processing()
            
        param_info['value'].trace_add('write', update_label)
        update_label()  # Initialize label

    def toggle_filter(self, filter_name):
        filter_state = self.filters[filter_name]
        if filter_state.enabled.get():
            filter_state.slider_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            filter_state.slider_frame.pack_forget()
        self.update_processing()

    def create_image_displays(self, parent):
        # Create frames for images
        self.original_frame = ttk.LabelFrame(parent, text="Original")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_frame = ttk.LabelFrame(parent, text="Processed")
        self.processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create canvases
        self.original_canvas = tk.Canvas(self.original_frame, bg='gray')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.processed_canvas = tk.Canvas(self.processed_frame, bg='gray')
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
    def select_directory(self):
        self.current_directory = filedialog.askdirectory()
        if self.current_directory:
            self.image_files = [f for f in os.listdir(self.current_directory)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if self.image_files:
                self.current_image_index = 0
                self.load_current_image()

    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            image_path = os.path.join(self.current_directory, 
                                    self.image_files[self.current_image_index])
            self.original_image = cv2.imread(image_path)
            if self.original_image is not None:
                self.update_processing()

    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()

    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

    def apply_filters(self, image):
        if image is None:
            return None

        # Convert to grayscale if not already
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image.copy()

        # Apply enabled filters in sequence
        if self.filters['denoise'].enabled.get():
            h = self.filters['denoise'].params['h']['value'].get()
            processed = cv2.fastNlMeansDenoising(processed, None, h=h)

        if self.filters['gaussian'].enabled.get():
            ksize = int(self.filters['gaussian'].params['kernel_size']['value'].get())
            sigma = self.filters['gaussian'].params['sigma']['value'].get()
            if ksize % 2 == 0:
                ksize += 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), sigma)

        if self.filters['bilateral'].enabled.get():
            d = int(self.filters['bilateral'].params['d']['value'].get())
            sigma_color = self.filters['bilateral'].params['sigma_color']['value'].get()
            sigma_space = self.filters['bilateral'].params['sigma_space']['value'].get()
            processed = cv2.bilateralFilter(processed, d, sigma_color, sigma_space)

        if self.filters['brightness_contrast'].enabled.get():
            brightness = self.filters['brightness_contrast'].params['brightness']['value'].get()
            contrast = self.filters['brightness_contrast'].params['contrast']['value'].get()
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)

        if self.filters['gamma'].enabled.get():
            gamma = self.filters['gamma'].params['gamma']['value'].get()
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            processed = cv2.LUT(processed, lookUpTable)

        if self.filters['sharpen'].enabled.get():
            amount = self.filters['sharpen'].params['amount']['value'].get()
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * amount
            processed = cv2.filter2D(processed, -1, kernel)

        if self.filters['clahe'].enabled.get():
            clip_limit = self.filters['clahe'].params['clip_limit']['value'].get()
            grid_size = int(self.filters['clahe'].params['grid_size']['value'].get())
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            processed = clahe.apply(processed)

        if self.filters['morphology'].enabled.get():
            ksize = int(self.filters['morphology'].params['kernel_size']['value'].get())
            kernel = np.ones((ksize, ksize), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        if self.filters['canny'].enabled.get():
            t1 = self.filters['canny'].params['threshold1']['value'].get()
            t2 = self.filters['canny'].params['threshold2']['value'].get()
            processed = cv2.Canny(processed, t1, t2)

        if self.filters['threshold'].enabled.get():
            block_size = int(self.filters['threshold'].params['block_size']['value'].get())
            C = self.filters['threshold'].params['C']['value'].get()
            if block_size % 2 == 0:
                block_size += 1
            processed = cv2.adaptiveThreshold(processed, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY,
                                           block_size, C)

        return processed

    def update_processing(self, *args):
        if self.original_image is not None:
            # Process image
            self.processed_image = self.apply_filters(self.original_image)
            # Update displays
            self.update_image_displays()

    def update_image_displays(self):
        if self.original_image is None:
            return

        # Get canvas sizes
        original_width = self.original_canvas.winfo_width()
        original_height = self.original_canvas.winfo_height()
        processed_width = self.processed_canvas.winfo_width()
        processed_height = self.processed_canvas.winfo_height()

        if min(original_width, original_height, processed_width, processed_height) < 1:
            self.root.after(100, self.update_image_displays)
            return

        # Convert original image for display
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)

        # Convert processed image for display
        processed_pil = Image.fromarray(self.processed_image)

        # Calculate scaling factors
        original_scale = min(original_width/original_pil.width, 
                           original_height/original_pil.height)
        processed_scale = min(processed_width/processed_pil.width, 
                            processed_height/processed_pil.height)

        # Resize images
        new_size = (
            int(original_pil.width * original_scale),
            int(original_pil.height * original_scale)
        )
        original_pil = original_pil.resize(new_size, Image.LANCZOS)
        processed_pil = processed_pil.resize(new_size, Image.LANCZOS)

        # Convert to PhotoImage
        self.original_photo = ImageTk.PhotoImage(original_pil)
        self.processed_photo = ImageTk.PhotoImage(processed_pil)

        # Update canvases
        self.original_canvas.delete("all")
        self.processed_canvas.delete("all")

        # Center images in canvases
        self.original_canvas.create_image(original_width//2, original_height//2,
                                        image=self.original_photo, anchor="center")
        self.processed_canvas.create_image(processed_width//2, processed_height//2,
                                         image=self.processed_photo, anchor="center")

    def get_current_filter_params(self):
        """Capture current filter parameters"""
        params = {}
        for filter_name, filter_state in self.filters.items():
            params[filter_name] = {
                'enabled': filter_state.enabled.get(),
                **{name: param['value'].get() 
                for name, param in filter_state.params.items()}
            }
        return params
    def process_single_image(self, image_file, input_dir, output_dir, filter_params):
        """Process a single image with given parameters"""
        image_path = os.path.join(input_dir, image_file)
        try:
            img = cv2.imread(image_path)
            if img is None:
                return f"Error reading {image_file}"

            # Convert to grayscale
            if len(img.shape) == 3:
                processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                processed = img.copy()

            # Apply filters based on saved parameters
            if filter_params['denoise']['enabled']:
                h = filter_params['denoise']['h']
                processed = cv2.fastNlMeansDenoising(processed, None, h=h)

            if filter_params['gaussian']['enabled']:
                ksize = int(filter_params['gaussian']['kernel_size'])
                if ksize % 2 == 0:
                    ksize += 1
                sigma = filter_params['gaussian']['sigma']
                processed = cv2.GaussianBlur(processed, (ksize, ksize), sigma)

            if filter_params['bilateral']['enabled']:
                d = int(filter_params['bilateral']['d'])
                sigma_color = filter_params['bilateral']['sigma_color']
                sigma_space = filter_params['bilateral']['sigma_space']
                processed = cv2.bilateralFilter(processed, d, sigma_color, sigma_space)

            if filter_params['brightness_contrast']['enabled']:
                brightness = filter_params['brightness_contrast']['brightness']
                contrast = filter_params['brightness_contrast']['contrast']
                processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)

            if filter_params['gamma']['enabled']:
                gamma = filter_params['gamma']['gamma']
                lookUpTable = np.empty((1,256), np.uint8)
                for i in range(256):
                    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
                processed = cv2.LUT(processed, lookUpTable)

            if filter_params['sharpen']['enabled']:
                amount = filter_params['sharpen']['amount']
                kernel = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]]) * amount
                processed = cv2.filter2D(processed, -1, kernel)

            if filter_params['clahe']['enabled']:
                clip_limit = filter_params['clahe']['clip_limit']
                grid_size = int(filter_params['clahe']['grid_size'])
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                processed = clahe.apply(processed)

            if filter_params['morphology']['enabled']:
                ksize = int(filter_params['morphology']['kernel_size'])
                kernel = np.ones((ksize, ksize), np.uint8)
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

            if filter_params['canny']['enabled']:
                t1 = filter_params['canny']['threshold1']
                t2 = filter_params['canny']['threshold2']
                processed = cv2.Canny(processed, t1, t2)

            if filter_params['threshold']['enabled']:
                block_size = int(filter_params['threshold']['block_size'])
                if block_size % 2 == 0:
                    block_size += 1
                C = filter_params['threshold']['C']
                processed = cv2.adaptiveThreshold(processed, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            block_size, C)
                # Invert for better OCR
                processed = cv2.bitwise_not(processed)

            # Save processed image
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, processed)
            return f"Successfully processed {image_file}"
        except Exception as e:
            return f"Error processing {image_file}: {str(e)}"
    
    def batch_process_directory(self):
        if not self.current_directory:
            tk.messagebox.showerror("Error", "Please select an input directory first")
            return
            
        if not self.output_base_directory:
            tk.messagebox.showerror("Error", "Please select an output base directory first")
            return

        import queue
        import threading

        # Create output directory with input directory name
        input_dir_name = os.path.basename(os.path.normpath(self.current_directory))
        output_dir = os.path.join(self.output_base_directory, f"{input_dir_name}_processed")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Capture current filter parameters
        filter_params = self.get_current_filter_params()

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Images")
        progress_window.geometry("400x200")
        progress_window.transient(self.root)
        progress_window.grab_set()

        # Progress display
        progress_frame = ttk.Frame(progress_window, padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True)

        progress_label = ttk.Label(progress_frame, text="Processing images...")
        progress_label.pack(pady=5)

        progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        progress_bar.pack(pady=5)

        status_text = tk.Text(progress_frame, height=5, width=40)
        status_text.pack(pady=5)

        # Create queues
        result_queue = queue.Queue()
        
        # Shared variables
        total_images = len(self.image_files)
        processed_count = 0
        should_stop = threading.Event()

        def worker():
            while not should_stop.is_set():
                try:
                    image_file = work_queue.get_nowait()
                    result = self.process_single_image(image_file, self.current_directory, 
                                                    output_dir, filter_params)
                    result_queue.put(result)
                    work_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    result_queue.put(f"Error processing {image_file}: {str(e)}")
                    work_queue.task_done()

        def update_progress():
            nonlocal processed_count
            
            try:
                while True:  # Process all available results
                    result = result_queue.get_nowait()
                    status_text.insert(tk.END, result + '\n')
                    status_text.see(tk.END)
                    processed_count += 1
                    progress_bar['value'] = (processed_count / total_images) * 100
                    result_queue.task_done()
            except queue.Empty:
                pass

            if processed_count < total_images:
                # Schedule next update
                progress_window.after(100, update_progress)
            else:
                # All done
                should_stop.set()
                progress_window.destroy()
                tk.messagebox.showinfo(
                    "Complete",
                    f"Processing complete!\nProcessed {total_images} images\nSaved to: {output_dir}"
                )

        # Create and populate work queue
        work_queue = queue.Queue()
        for image_file in self.image_files:
            work_queue.put(image_file)

        # Start worker threads
        num_threads = max(1, mp.cpu_count() - 1)
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)

        # Start progress updates
        progress_window.after(100, update_progress)
def main():
    root = tk.Tk()
    root.title("Image Processor")
    root.geometry("1200x800")
    app = ImageProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()