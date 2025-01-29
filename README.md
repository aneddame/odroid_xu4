# Object Detection and Similarity Computation

## Project Overview

This project involves the implementation of object detection and fuzzy similarity computation between two images. It uses a MobileNet-SSD model for object detection and a Sobel filter to compute image similarity based on the structure of the images.

The code utilizes OpenCL for efficient parallel processing on the GPU, OpenMP for multi-threading, and OpenCV for image processing. The Fuzzy Longest Common Subsequence Similarity (F_LCSS) metric is used to compare the detected objects between two images.

### Key Technologies:
- **OpenCL**: For GPU-accelerated image processing with the Mali-T628 GPU.
- **OpenMP**: For parallel programming using multi-core processors.
- **OpenCV**: For image manipulation and object detection.
- **Deep Learning**: MobileNet-SSD for object detection.

## Project Features

1. **Object Detection**: Using MobileNet-SSD (Caffe model), the code detects objects in images and highlights them with bounding boxes and class labels.
   
2. **Sobel Filter**: A Sobel filter is applied to the images to compute edge information, which is used for image similarity comparison.

3. **Fuzzy Similarity (F_LCSS)**: This metric computes the similarity between two images based on their Sobel-filtered outputs.

4. **OpenCL and OpenMP**: The project uses OpenCL for GPU acceleration and OpenMP for multi-core CPU processing, providing better performance for the computationally expensive parts.

## Setup and Requirements

To run this project, you need the following:

- **C++**: For compiling and running the main program.
- **OpenCV**: For image processing and object detection.
- **OpenCL**: For GPU-accelerated computations.
- **OpenMP**: For parallel CPU processing.
- **Caffe Model Files**: MobileNet-SSD model files (`.prototxt` and `.caffemodel`).

### Installation

1. **Install OpenCV**: Follow the official OpenCV installation guide for your platform.
2. **Install OpenCL**: Install the OpenCL SDK suitable for your GPU (Mali-T628 in this case).
3. **Download Caffe Model**: You will need the MobileNet-SSD model files. Download them from [here](https://github.com/chuanqi305/MobileNet-SSD).

### Build Instructions

1. Clone the repository to your local machine:

   ```bash
   git clone <your-repository-url>
