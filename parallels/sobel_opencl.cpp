// g++ -std=c++11 -o sobel_filter sobel_filter.cpp -lOpenCL `pkg-config --cflags --libs opencv4`

#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

// Macro for error checking
#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "%s failed with error %d\n", msg, err); \
        exit(1); \
    }

// OpenCL Sobel filter kernel source code
const char* kernelSource = R"CLC(
__kernel void sobel_filter(__global const uchar* input, __global uchar* output, const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = 0;
        int gy = 0;

        // Sobel kernels
        const int sobelX[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        const int sobelY[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int pixel = input[(y + ky) * width + (x + kx)];
                gx += pixel * sobelX[ky + 1][kx + 1];
                gy += pixel * sobelY[ky + 1][kx + 1];
            }
        }

        int magnitude = sqrt((float)(gx * gx + gy * gy));
        output[y * width + x] = (uchar)clamp(magnitude, 0, 255);
    }
}
)CLC";

// Function to display OpenCL build log
void checkBuildLog(cl_program program, cl_device_id device) {
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

    char* buildLog = (char*)malloc(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
    fprintf(stderr, "Build log:\n%s\n", buildLog);
    free(buildLog);
}

int main() {
    const char* imagePath = "/home/odroid/Desktop/flcss/img/right/3.jpeg";

    // Load the input image using OpenCV
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        fprintf(stderr, "Failed to load image: %s\n", imagePath);
        return -1;
    }

    // Resize the input image
    int newWidth = 512;  // Set desired width
    int newHeight = 512; // Set desired height
    cv::resize(inputImage, inputImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    int width = inputImage.cols;
    int height = inputImage.rows;
    size_t imageSize = width * height;

    // Create an output image
    cv::Mat outputImage(height, width, CV_8UC1);

    cl_int err;

    // Get platform and device
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "Getting platform");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "Getting device");

    // Create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Creating context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err, "Creating command queue");

    // Create memory buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &err);
    CHECK_ERROR(err, "Creating input buffer");

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, NULL, &err);
    CHECK_ERROR(err, "Creating output buffer");

    // Copy input image to device
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, imageSize, inputImage.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Writing input image to device");

    // Create and build OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_ERROR(err, "Creating program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building program\n");
        checkBuildLog(program, device);
        exit(1);
    }

    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "sobel_filter", &err);
    CHECK_ERROR(err, "Creating kernel");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    CHECK_ERROR(err, "Setting kernel arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    CHECK_ERROR(err, "Setting kernel arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(int), &width);
    CHECK_ERROR(err, "Setting kernel arg 2");

    err = clSetKernelArg(kernel, 3, sizeof(int), &height);
    CHECK_ERROR(err, "Setting kernel arg 3");

    // Execute the kernel
    size_t globalSize[2] = { (size_t)width, (size_t)height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "Enqueueing kernel");

    // Read back the result
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageSize, outputImage.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Reading output buffer");

    // Display the output image
    cv::imshow("Sobel Output", outputImage);
    cv::waitKey(0); // Wait for a key press to close the window

    printf("Sobel filter applied successfully. Output displayed.\n");

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
