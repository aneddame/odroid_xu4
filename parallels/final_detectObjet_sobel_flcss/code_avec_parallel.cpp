#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Définition des classes MobileNet-SSD
string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "%s failed with error %d\n", msg, err); \
        exit(1); \
    }

// Define Sobel filter kernel (optimized for Mali-T628 GPU)
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

// Charger le modèle MobileNet-SSD
Net loadModel() {
    String modelTxt = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/MobileNetSSD_deploy.prototxt";
    String modelBin = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/MobileNetSSD_deploy.caffemodel";

    Net net = readNetFromCaffe(modelTxt, modelBin);
    if (net.empty()) {
        cerr << "Erreur lors du chargement du modèle !" << endl;
        exit(-1);
    }
    return net;
}

// Détection d'objets avec OpenMP
Mat detectObjects(Net &net, Mat &img) {
    Mat img2;
    resize(img, img2, Size(300, 300));  // Resize to a smaller size to improve performance

    Mat inputBlob = blobFromImage(img2, 0.007843, Size(224, 224), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inputBlob, "data");

    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    float confidenceThreshold = 0.2;

    // Parallelize the loop using OpenMP with reduced threads
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold) {
            int idx = static_cast<int>(detectionMat.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            // Critical section to avoid race conditions when drawing on the image
            #pragma omp critical
            {
                rectangle(img, Rect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom), Scalar(0, 255, 0), 2);
                putText(img, CLASSES[idx] + " " + to_string(confidence), Point(xLeftBottom, yLeftBottom - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            }
        }
    }
    return img;
}

// Calculer la similarité F_LCSS
float F_LCSS(const Mat &sobel1, const Mat &sobel2) {
    int width = sobel1.cols;
    int height = sobel1.rows;
    float similarity = 0.0f;
    int totalPixels = width * height;

    #pragma omp parallel for reduction(+:similarity) num_threads(4)  // Reduced number of threads for better performance
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float diff = fabs(sobel1.at<uchar>(i, j) - sobel2.at<uchar>(i, j));
            float fuzzySim = exp(-diff);
            similarity += fuzzySim;
        }
    }

    return similarity / totalPixels;
}

int main() {
    double startTime = (double)getTickCount();

    // Charger les images
    string imageFile1 = "/home/odroid/Desktop/parallel/lidar/data/image_03/data/0000000063.png";
    string imageFile2 = "/home/odroid/Desktop/parallel/lidar/data/image_02/data/0000000063.png";

    Mat img1 = imread(imageFile1, IMREAD_COLOR);
    Mat img2 = imread(imageFile2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        cerr << "Impossible de charger les images !" << endl;
        return -1;
    }

    resize(img1, img1, Size(300, 300));  // Resize to a smaller size
    resize(img2, img2, Size(300, 300));  // Resize to a smaller size

    // Charger MobileNet-SSD et détecter les objets
    Net net = loadModel();
    Mat img1Detected = detectObjects(net, img1);
    Mat img2Detected = detectObjects(net, img2);

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
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, &event);
    CHECK_ERROR(err, "Enqueueing kernel");

    clWaitForEvents(1, &event);  // Asynchronous execution

    // Read back the output
    Mat sobelMag1(height, width, CV_8U);
    err = clEnqueueReadBuffer(queue, outputBuffer1, CL_TRUE, 0, imageSize, sobelMag1.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Reading output buffer");

    // Repeat for the second image
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer2);
    CHECK_ERROR(err, "Setting kernel arg for image 2");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer2);
    CHECK_ERROR(err, "Setting kernel arg for output 2");

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, &event);
    CHECK_ERROR(err, "Enqueueing kernel for second image");

    clWaitForEvents(1, &event);  // Asynchronous execution

    // Read back the output for the second image
    Mat sobelMag2(height, width, CV_8U);
    err = clEnqueueReadBuffer(queue, outputBuffer2, CL_TRUE, 0, imageSize, sobelMag2.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Reading output buffer");

    // Compute similarity between Sobel images
    float similarity = F_LCSS(sobelMag1, sobelMag2);
    cout << "Fuzzy LCSS Similarity: " << similarity << endl;

    // Calculate the percentage similarity
    float similarityPercentage = similarity * 100;
    cout << "Similarity Percentage: " << similarityPercentage << "%" << endl;

    double endTime = (double)getTickCount();
    double elapsedTime = (endTime - startTime) / getTickFrequency();
    cout << "Total Processing Time: " << elapsedTime << " seconds" << endl;

    // Feature tracking
    vector<Point2f> traj1, traj2;
    goodFeaturesToTrack(gray1, traj1, 100, 0.01, 10);
    goodFeaturesToTrack(gray2, traj2, 100, 0.01, 10);

    Mat img1Copy = img1.clone();
    Mat img2Copy = img2.clone();
    for (size_t i = 0; i < traj1.size() && i < traj2.size(); ++i) {
        circle(img1Copy, traj1[i], 5, Scalar(0, 0, 255), -1);  // Points rouges sur img1
        circle(img2Copy, traj2[i], 5, Scalar(255, 0, 0), -1);  // Points bleus sur img2
    }

    // Combine the images side by side
    Mat img_matches;
    hconcat(img1Copy, img2Copy, img_matches);

    // Show the combined image with matched points
    imshow("Detected Object Matches", img_matches);
    waitKey(0);

    // Cleanup OpenCL resources
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
