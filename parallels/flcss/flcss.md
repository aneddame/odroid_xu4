# Fuzzy Longest Common Subsequence (F_LCSS) with Sobel Edge Detection

This project applies **Fuzzy Longest Common Subsequence (F_LCSS)** to compare the edges detected in two images using **Sobel filters**. The approach first detects the edges in both images, extracts the points representing strong edges, and then compares these points using a fuzzy similarity function to compute a similarity score.

## Description

1. **Sobel Edge Detection**: The Sobel filter is applied manually to compute the gradients in both the horizontal and vertical directions. This step highlights the edges in the image by calculating the gradient magnitude at each pixel.

2. **Fuzzy Longest Common Subsequence (F_LCSS)**: After detecting edges in both images, the points where strong edges are detected are extracted as trajectories. These trajectories are then compared using a fuzzy similarity function. The function assigns a membership value to each point in the sequence based on its distance from other points. The F_LCSS score is calculated as a similarity percentage between the two image trajectories.

3. **Visualization**: The points corresponding to strong edges are marked on the images in different colors (red for image 1, blue for image 2). The images are then displayed side by side for easy visual comparison.

4. **Processing Time**: The program measures and outputs the time taken for each significant step, including image loading, Sobel edge detection, and F_LCSS computation.

## Requirements

- **OpenCV** (version 4.x or higher)
- A C++ compiler with support for C++11 or higher

## How to Run

1. Clone this repository to your local machine.
2. Install OpenCV for C++ (if not already installed).
3. Compile and run the program with the following command:

   ```bash
   g++ -std=c++11 -o fuzzy_lcss fuzzy_lcss.cpp `pkg-config --cflags --libs opencv4`
   ./fuzzy_lcss
