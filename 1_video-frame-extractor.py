import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_frame(args):
    """Extract a single frame - this will run in a separate process"""
    try:
        video_path, output_dir, frame_index, time_point, fps = args
        
        # Create a new video capture object for each process
        cap = cv2.VideoCapture(video_path)
        
        # Set frame position
        frame_position = int(time_point * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            return frame_index, False, f"Could not read frame at {time_point}s"
        
        # Save frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        cap.release()
        return frame_index, True, None
        
    except Exception as e:
        return frame_index, False, str(e)

class VideoFrameExtractor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Frame Extractor")
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        
        self.video_path = None
        self.video_duration = None
        self.num_processes = max(1, cpu_count() - 1)  # Leave one CPU core free
        
        self.create_widgets()
        
    def create_widgets(self):
        # Configure grid weight
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Video Selection", padding="10")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        file_frame.grid_columnconfigure(0, weight=1)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state='readonly')
        self.file_entry.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.select_video_file)
        self.browse_button.grid(row=0, column=1)
        
        # Duration settings section
        duration_frame = ttk.LabelFrame(main_frame, text="Duration Settings", padding="10")
        duration_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        duration_frame.grid_columnconfigure(1, weight=1)
        
        # Duration input
        ttk.Label(duration_frame, text="Duration (seconds):").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.duration_var = tk.StringVar(value="1.0")
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var)
        self.duration_entry.grid(row=0, column=1, sticky="ew", pady=2)
        
        # Interval input
        ttk.Label(duration_frame, text="Interval (seconds):").grid(row=1, column=0, sticky="w", padx=(0, 5))
        self.interval_var = tk.StringVar(value="0.1")
        self.interval_entry = ttk.Entry(duration_frame, textvariable=self.interval_var)
        self.interval_entry.grid(row=1, column=1, sticky="ew", pady=2)
        
        # Video information section
        self.info_frame = ttk.LabelFrame(main_frame, text="Video Information", padding="10")
        self.info_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        self.info_text = tk.StringVar(value="Select a video file to see information")
        self.info_label = ttk.Label(self.info_frame, textvariable=self.info_text)
        self.info_label.grid(row=0, column=0, sticky="w")
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, sticky="w")
        
        # Extract button
        self.extract_button = ttk.Button(main_frame, text="Extract Frames", command=self.start_extraction)
        self.extract_button.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.extract_button['state'] = 'disabled'
        
    def select_video_file(self):
        """Opens a file dialog to select a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self.extract_button['state'] = 'normal'
            self.update_video_info()
            
    def update_video_info(self):
        """Updates video information display"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = total_frames / fps
            
            duration_str = str(timedelta(seconds=int(self.video_duration)))
            info_text = f"Duration: {duration_str}\nFPS: {fps:.2f}\nTotal Frames: {total_frames}"
            
            self.info_text.set(info_text)
            cap.release()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read video information: {str(e)}")
            
    def validate_inputs(self):
        """Validates user inputs"""
        try:
            duration = float(self.duration_var.get())
            interval = float(self.interval_var.get())
            
            if duration <= 0 or interval <= 0:
                messagebox.showerror("Error", "Duration and interval must be positive numbers!")
                return False
                
            if duration > self.video_duration:
                if messagebox.askyesno("Warning", 
                    f"Requested duration ({duration}s) is longer than video duration ({self.video_duration:.2f}s).\n"
                    "Do you want to use the maximum video duration instead?"):
                    self.duration_var.set(str(self.video_duration))
                return False
                
            return True
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for duration and interval!")
            return False
            
    
    def extract_frames(self):
        """Extracts frames from the video using multiple processes"""
        try:
            duration = float(self.duration_var.get())
            interval = float(self.interval_var.get())
            
            # Create output directory
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_dir = f"frames_{video_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Get video properties
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Generate frame extraction tasks
            frame_times = []
            current_time = 0
            frame_index = 0
            
            while current_time <= duration:
                frame_times.append((
                    self.video_path,
                    output_dir,
                    frame_index,
                    current_time,
                    fps
                ))
                current_time += interval
                frame_index += 1
            
            total_frames = len(frame_times)
            processed_frames = 0
            errors = []

            # Create process pool and start extraction
            # Note: extract_frame is now a standalone function, not a method
            with Pool(processes=self.num_processes) as pool:
                for frame_index, success, error in pool.imap_unordered(extract_frame, frame_times):
                    processed_frames += 1
                    progress = (processed_frames / total_frames) * 100
                    
                    if not success:
                        errors.append(f"Frame {frame_index}: {error}")
                    
                    # Update progress bar and status
                    self.progress_bar['value'] = progress
                    self.status_var.set(f"Processing: {processed_frames}/{total_frames} frames ({progress:.1f}%)")
                    self.root.update()

            # Show completion message and any errors
            self.progress_bar['value'] = 100
            if errors:
                error_msg = "\n".join(errors)
                messagebox.showwarning("Completed with errors", 
                    f"Frame extraction completed with {len(errors)} errors:\n\n{error_msg}")
            else:
                self.status_var.set(f"Extraction complete! {total_frames} frames saved to '{output_dir}' directory")
                messagebox.showinfo("Complete", "Frame extraction completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.extract_button['state'] = 'normal'
            self.browse_button['state'] = 'normal'
            
    def start_extraction(self):
        """Starts the frame extraction process"""
        if self.validate_inputs():
            self.extract_button['state'] = 'disabled'
            self.browse_button['state'] = 'disabled'
            self.extract_frames()
            
    def run(self):
        """Starts the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoFrameExtractor()
    app.run()
