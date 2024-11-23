# Dashboard Video Processing Pipeline

A comprehensive toolkit for extracting and processing time series data from dashboard video recordings. This pipeline consists of multiple Python scripts that work together to extract frames, process regions of interest, and perform OCR to generate time series data.

## Overview

This toolkit provides a complete workflow for:
1. Extracting frames from video recordings
2. Processing specific regions of interest (ROI) from images
3. Advanced image processing for text/number recognition
4. OCR-based data extraction to generate time series

## Components

### 1. Video Frame Extractor (`1_video-frame-extractor.py`)
- GUI application for extracting frames from video files
- Features:
  - Configurable frame extraction interval
  - Multi-processing support for faster extraction
  - Progress tracking
  - Custom output directory selection

### 2. ROI Cropper (`2_Image_roi_Croper.py`)
- GUI tool for selecting and cropping regions of interest from images
- Features:
  - Visual ROI selection interface
  - Batch processing capabilities
  - Multi-processing support
  - ROI coordinate saving/loading

### 3. Image Processor (`3_Image_processor.py`)
- Advanced image processing tool with multiple filters and adjustments
- Features:
  - Multiple image processing filters (Gaussian, Bilateral, CLAHE, etc.)
  - Real-time preview
  - Batch processing
  - Parameter adjustment interface
  - Customizable processing pipeline

### 4. OCR Data Extractor (`4_OCR_data_extraction.py`)
- Specialized tool for extracting numerical data using OCR
- Features:
  - EasyOCR integration
  - Advanced image preprocessing
  - Density-based analysis
  - Batch processing capability
  - CSV output generation

## Installation

### Prerequisites
- Python 3.7 or higher
- Git
- On Linux systems: basic development tools and libraries

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dashboard-video-processor.git
cd dashboard-video-processor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Run the prerequisite installer:
```bash
python prerequisite-installer.py
```

The prerequisite installer will automatically:
- Verify Python version compatibility
- Install and configure all required dependencies in the correct order
- Handle package conflicts (especially OpenCV-related issues)
- Verify GUI support
- Install system-level dependencies (on Linux)

Required packages that will be installed include:
- EasyOCR for text recognition
- OpenCV (with GUI support) for image processing
- NumPy for numerical operations
- Pillow for image handling
- Tkinter for GUI (usually comes with Python)

### Manual Installation (if needed)

If you prefer to install dependencies manually or if the prerequisite installer encounters issues:

1. Install base dependencies:
```bash
pip install setuptools wheel
```

2. Install core packages:
```bash
pip install easyocr numpy pillow opencv-python
```

### System-Specific Notes

#### Linux
The prerequisite installer will attempt to install required system packages:
- python3-tk
- python3-pil
- python3-pil.imagetk
- libgl1-mesa-glx

If you're installing manually, install these using your distribution's package manager:
```bash
sudo apt-get update
sudo apt-get install python3-tk python3-pil python3-pil.imagetk libgl1-mesa-glx
```

#### Windows
- Ensure you have the Visual C++ redistributable installed (required for OpenCV)
- If using Anaconda, some packages may need to be installed through conda instead of pip

#### macOS
- Tkinter should be included with Python
- You may need to install Command Line Tools for Xcode for some compilations

## Workflow

### Step 1: Frame Extraction
1. Run the frame extractor:
```bash
python 1_video-frame-extractor.py
```
2. Select your video file
3. Configure extraction parameters:
   - Duration (in seconds)
   - Interval between frames
4. Select output directory
5. Start extraction

### Step 2: ROI Selection
1. Run the ROI selector:
```bash
python 2_Image_roi_Croper.py
```
2. Load a sample frame
3. Draw rectangles around regions of interest
4. Save ROI coordinates
5. Batch process all extracted frames

### Step 3: Image Processing
1. Run the image processor:
```bash
python 3_Image_processor.py
```
2. Load cropped images
3. Configure processing parameters:
   - Adjust filters
   - Set thresholds
   - Configure preprocessing options
4. Batch process images

### Step 4: Data Extraction
1. Run the OCR extractor:
```bash
python 4_OCR_data_extraction.py
```
2. Load processed images
3. Configure OCR parameters:
   - Set confidence thresholds
   - Adjust density parameters
4. Run batch extraction
5. Export results to CSV

## Usage Tips

### Frame Extraction
- Choose an appropriate interval based on your video's update frequency
- Use the preview feature to verify frame quality
- Consider storage space when selecting extraction interval

### ROI Selection
- Select a clear, representative frame for ROI definition
- Add padding around numbers to account for slight movements
- Use the debug mode for precise selection

### Image Processing
- Start with default parameters and adjust as needed
- Use the real-time preview to verify filter effects
- Save successful configurations for future use

### OCR Extraction
- Verify OCR accuracy on a few sample images before batch processing
- Use debug mode to visualize detection regions
- Adjust confidence thresholds based on image quality

## Output Format

The final output is a CSV file containing:
- Timestamp
- Extracted numerical values
- Source frame reference



## Acknowledgments

- EasyOCR for text recognition
- OpenCV for image processing
- The Python community for various supporting libraries

---
<sub>
Note: Parts of this codebase were developed with assistance from Claude (Anthropic) AI, particularly for code structure, error handling, and GUI implementations. Core algorithms and logic were independently developed.
</sub>
