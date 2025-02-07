#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <dirent.h> // For directory handling

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

// Function to get all image files in a directory
vector<string> getImageFiles(const string& folderPath) {
    vector<string> imageFiles;
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string fileName = ent->d_name;
            if (fileName.find(".jpg") != string::npos || fileName.find(".png") != string::npos) {
                imageFiles.push_back(folderPath + "/" + fileName);
            }
        }
        closedir(dir);
    } else {
        cerr << "Could not open directory: " << folderPath << endl;
        exit(-1);
    }
    return imageFiles;
}

int main() {
    double startTime = (double)getTickCount();

    // Path to the dataset folder
    string folderPath = "/home/odroid/Desktop/parallel/lidar/data/image_02/data/";

    // Get all image files in the folder
    vector<string> imageFiles = getImageFiles(folderPath);
    if (imageFiles.empty()) {
        cerr << "No images found in the folder!" << endl;
        return -1;
    }

    // Charger MobileNet-SSD
    Net net = loadModel();

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_command_queue queue;
    cl_context context = setup_opencl(device, queue, platform);

    // Process each image in the dataset
    for (const string& imageFile : imageFiles) {
        Mat img = imread(imageFile, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Impossible de charger l'image: " << imageFile << endl;
            continue;
        }

        // Convertir en niveaux de gris
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        int width = gray.cols;
        int height = gray.rows;
        size_t imageSize = width * height;

        // Create OpenCL buffers
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, NULL);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);

        cl_int err;
        err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, imageSize, gray.data, 0, NULL, NULL);
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

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        CHECK_ERROR(err, "Setting kernel arg 0");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
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
        Mat sobelMag(height, width, CV_8U);
        err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageSize, sobelMag.data, 0, NULL, NULL);
        CHECK_ERROR(err, "Reading output buffer");

        // Détecter les objets dans l'image originale (pas dans l'image Sobel)
        Mat imgDetected = detectObjects(net, img);

        // Afficher uniquement l'image avec les objets détectés (bounding boxes)
        imshow("Detected Objects", imgDetected);
        waitKey(1); // Wait for a short time to display the image

        // Cleanup OpenCL resources for this image
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }

    // Cleanup OpenCL resources
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    double endTime = (double)getTickCount();
    double elapsedTime = (endTime - startTime) / getTickFrequency();
    cout << "Total Processing Time: " << elapsedTime << " seconds" << endl;

    return 0;
}
