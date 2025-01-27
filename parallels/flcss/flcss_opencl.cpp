#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <ctime>  // For time measurement

using namespace cv;
using namespace std;

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "%s failed with error %d\n", msg, err); \
        exit(1); \
    }

// Define Sobel filter kernel (updated)
const char* kernelSource = R"CLC(
__kernel void sobel_filter(__global const uchar* input, __global uchar* output, const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = 0;
        int gy = 0;

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

// Function to set up OpenCL environment
cl_context setup_opencl(cl_device_id &device, cl_command_queue &queue, cl_platform_id &platform) {
    cl_int err;

    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "Getting platform");

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "Getting device");

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Creating context");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err, "Creating command queue");

    return context;
}

// Fuzzy Longest Common Subsequence (F_LCSS) computation
float F_LCSS(const Mat &sobelMag1, const Mat &sobelMag2) {
    int width = sobelMag1.cols;
    int height = sobelMag1.rows;

    float similarity = 0.0f;
    int totalPixels = width * height;

    // Fuzzy logic computation (example)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float diff = fabs(sobelMag1.at<uchar>(i, j) - sobelMag2.at<uchar>(i, j));
            float fuzzySim = exp(-diff); // Fuzzy similarity based on difference
            similarity += fuzzySim;
        }
    }

    return similarity / totalPixels;
}

int main() {
    double startTime = (double)getTickCount();

    // Load image files
    string imageFile1 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";
    string imageFile2 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";

    Mat img1 = imread(imageFile1, IMREAD_COLOR);
    Mat img2 = imread(imageFile2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        cerr << "Unable to load images." << endl;
        return -1;
    }

    resize(img1, img1, Size(300, 300));
    resize(img2, img2, Size(300, 300));

    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    int width = gray1.cols;
    int height = gray1.rows;
    size_t imageSize = width * height;

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_command_queue queue;
    cl_context context = setup_opencl(device, queue, platform);

    // Create OpenCL buffers
    cl_mem inputBuffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, NULL);
    cl_mem inputBuffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, NULL);
    cl_mem outputBuffer1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);
    cl_mem outputBuffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);

    cl_int err;
    err = clEnqueueWriteBuffer(queue, inputBuffer1, CL_TRUE, 0, imageSize, gray1.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Writing input buffer");

    err = clEnqueueWriteBuffer(queue, inputBuffer2, CL_TRUE, 0, imageSize, gray2.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Writing input buffer");

    // Create program and compile
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_ERROR(err, "Creating program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        cerr << "Error building program." << endl;
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "sobel_filter", &err);
    CHECK_ERROR(err, "Creating kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer1);
    CHECK_ERROR(err, "Setting kernel arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer1);
    CHECK_ERROR(err, "Setting kernel arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(int), &width);
    CHECK_ERROR(err, "Setting kernel arg 2");

    err = clSetKernelArg(kernel, 3, sizeof(int), &height);
    CHECK_ERROR(err, "Setting kernel arg 3");

    size_t globalSize[2] = { (size_t)width, (size_t)height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "Enqueueing kernel");

    clFinish(queue);

    // Read back the output
    Mat sobelMag1(height, width, CV_8U);
    err = clEnqueueReadBuffer(queue, outputBuffer1, CL_TRUE, 0, imageSize, sobelMag1.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Reading output buffer");

    // Repeat for the second image
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer2);
    CHECK_ERROR(err, "Setting kernel arg for image 2");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer2);
    CHECK_ERROR(err, "Setting kernel arg for output 2");

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "Enqueueing kernel for second image");

    clFinish(queue);

    Mat sobelMag2(height, width, CV_8U);
    err = clEnqueueReadBuffer(queue, outputBuffer2, CL_TRUE, 0, imageSize, sobelMag2.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Reading output buffer for image 2");

    // Calculate fuzzy similarity (F_LCSS)
    float similarity = F_LCSS(sobelMag1, sobelMag2);
    cout << "Fuzzy Similarity (F_LCSS) = " << similarity << endl;

    // Calculate similarity percentage
    float similarityPercentage = similarity * 100;
    cout << "Similarity Percentage = " << similarityPercentage << "%" << endl;

    // Calculate processing time
    double endTime = (double)getTickCount();
    double processingTime = (endTime - startTime) / getTickFrequency();
    cout << "Temps de traitement : " << processingTime << " secondes" << endl;

    // Afficher les correspondances des points détectés sur les images originales
    vector<Point2f> traj1, traj2;
    goodFeaturesToTrack(gray1, traj1, 100, 0.01, 10);
    goodFeaturesToTrack(gray2, traj2, 100, 0.01, 10);

    Mat img1Copy = img1.clone();
    Mat img2Copy = img2.clone();
    for (size_t i = 0; i < traj1.size() && i < traj2.size(); ++i) {
        circle(img1Copy, traj1[i], 5, Scalar(0, 0, 255), -1);  // Points rouges sur img1
        circle(img2Copy, traj2[i], 5, Scalar(255, 0, 0), -1);  // Points bleus sur img2
    }

    // Combiner les deux images originales côte à côte
    Mat img_matches;
    hconcat(img1Copy, img2Copy, img_matches);

    // Afficher les images combinées
    imshow("Correspondances des bords détectés", img_matches);
    waitKey(0);

    // Release OpenCL resources
    clReleaseMemObject(inputBuffer1);
    clReleaseMemObject(inputBuffer2);
    clReleaseMemObject(outputBuffer1);
    clReleaseMemObject(outputBuffer2);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
