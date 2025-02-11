#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
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

// OpenCL kernel for computing IoU
const char* iouKernelSource = R"(
__kernel void computeIoU(__global int* bbox1, __global int* bbox2, __global float* result, int numBboxes) {
    int gid = get_global_id(0);
    if (gid >= numBboxes) return;

    int x1 = max(bbox1[gid * 4 + 0], bbox2[gid * 4 + 0]);
    int y1 = max(bbox1[gid * 4 + 1], bbox2[gid * 4 + 1]);
    int x2 = min(bbox1[gid * 4 + 0] + bbox1[gid * 4 + 2], bbox2[gid * 4 + 0] + bbox2[gid * 4 + 2]);
    int y2 = min(bbox1[gid * 4 + 1] + bbox1[gid * 4 + 3], bbox2[gid * 4 + 1] + bbox2[gid * 4 + 3]);

    int intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    int bbox1Area = bbox1[gid * 4 + 2] * bbox1[gid * 4 + 3];
    int bbox2Area = bbox2[gid * 4 + 2] * bbox2[gid * 4 + 3];

    int unionArea = bbox1Area + bbox2Area - intersectionArea;

    result[gid] = (float)intersectionArea / (float)unionArea;
}
)";

// Function to compute Intersection over Union (IoU) on GPU
vector<float> computeIoU_GPU(cl_context context, cl_command_queue queue, cl_kernel kernel, const vector<Rect>& bboxes1, const vector<Rect>& bboxes2) {
    cl_int err;

    // Prepare data for GPU
    vector<int> bbox1Data, bbox2Data;
    for (const auto& bbox : bboxes1) {
        bbox1Data.push_back(bbox.x);
        bbox1Data.push_back(bbox.y);
        bbox1Data.push_back(bbox.width);
        bbox1Data.push_back(bbox.height);
    }
    for (const auto& bbox : bboxes2) {
        bbox2Data.push_back(bbox.x);
        bbox2Data.push_back(bbox.y);
        bbox2Data.push_back(bbox.width);
        bbox2Data.push_back(bbox.height);
    }

    size_t numBboxes = bboxes1.size();
    vector<float> results(numBboxes);

    // Validate buffer sizes
    size_t bbox1BufferSize = bbox1Data.size() * sizeof(int);
    size_t bbox2BufferSize = bbox2Data.size() * sizeof(int);
    size_t resultBufferSize = numBboxes * sizeof(float);

    if (bbox1BufferSize == 0 || bbox2BufferSize == 0 || resultBufferSize == 0) {
        cerr << "Error: Buffer size is zero. Check bounding box data." << endl;
        return results;
    }

    // Create OpenCL buffers
    cl_mem bbox1Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bbox1BufferSize, bbox1Data.data(), &err);
    if (err != CL_SUCCESS) {
        cerr << "clCreateBuffer bbox1Buffer failed with error " << err << ". Buffer size: " << bbox1BufferSize << endl;
        return results;
    }
    cl_mem bbox2Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bbox2BufferSize, bbox2Data.data(), &err);
    if (err != CL_SUCCESS) {
        cerr << "clCreateBuffer bbox2Buffer failed with error " << err << ". Buffer size: " << bbox2BufferSize << endl;
        clReleaseMemObject(bbox1Buffer);
        return results;
    }
    cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resultBufferSize, NULL, &err);
    if (err != CL_SUCCESS) {
        cerr << "clCreateBuffer resultBuffer failed with error " << err << ". Buffer size: " << resultBufferSize << endl;
        clReleaseMemObject(bbox1Buffer);
        clReleaseMemObject(bbox2Buffer);
        return results;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bbox1Buffer);
    CHECK_ERROR(err, "clSetKernelArg bbox1Buffer");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bbox2Buffer);
    CHECK_ERROR(err, "clSetKernelArg bbox2Buffer");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultBuffer);
    CHECK_ERROR(err, "clSetKernelArg resultBuffer");
    err = clSetKernelArg(kernel, 3, sizeof(int), &numBboxes);
    CHECK_ERROR(err, "clSetKernelArg numBboxes");

    // Execute the kernel
    size_t globalSize = numBboxes;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "clEnqueueNDRangeKernel");

    // Read the results back from the GPU
    err = clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0, resultBufferSize, results.data(), 0, NULL, NULL);
    CHECK_ERROR(err, "clEnqueueReadBuffer");

    // Cleanup
    clReleaseMemObject(bbox1Buffer);
    clReleaseMemObject(bbox2Buffer);
    clReleaseMemObject(resultBuffer);

    return results;
}

// Function to load timestamps from file
vector<string> loadTimestamps(const string& timestampFile) {
    vector<string> timestamps;
    ifstream file(timestampFile);
    if (!file.is_open()) {
        cerr << "Could not open timestamp file: " << timestampFile << endl;
        exit(-1);
    }
    string line;
    while (getline(file, line)) {
        timestamps.push_back(line);
    }
    file.close();
    return timestamps;
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
    // Sort files to ensure they are processed in order
    sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
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
vector<Rect> detectObjects(Net &net, Mat &img, vector<Point2f>& centers) {
    Mat img2;
    resize(img, img2, Size(300, 300));  // Resize to a smaller size to improve performance

    Mat inputBlob = blobFromImage(img2, 0.007843, Size(224, 224), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inputBlob, "data");

    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    float confidenceThreshold = 0.2;
    vector<Rect> bboxes;

    // Parallelize the loop using OpenMP with reduced threads
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold) {
            int idx = static_cast<int>(detectionMat.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            Rect bbox(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
            Point2f center((xLeftBottom + xRightTop) / 2, (yLeftBottom + yRightTop) / 2);

            #pragma omp critical
            {
                bboxes.push_back(bbox);
                centers.push_back(center);
                rectangle(img, bbox, Scalar(0, 255, 0), 2);
                putText(img, CLASSES[idx] + " " + to_string(confidence), Point(xLeftBottom, yLeftBottom - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
                circle(img, center, 3, Scalar(255, 0, 0), -1); // Draw center point
            }
        }
    }
    return bboxes;
}

int main() {
    double startTime = (double)getTickCount();

    // Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Get platform and device
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "clGetDeviceIDs");

    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err, "clCreateCommandQueue");

    // Create program from kernel source
    program = clCreateProgramWithSource(context, 1, &iouKernelSource, NULL, &err);
    CHECK_ERROR(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    CHECK_ERROR(err, "clBuildProgram");

    // Create kernel
    kernel = clCreateKernel(program, "computeIoU", &err);
    CHECK_ERROR(err, "clCreateKernel");

    // Query device limits
    cl_ulong maxMemAllocSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAllocSize, NULL);
    CHECK_ERROR(err, "clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE");
    cout << "Max memory allocation size: " << maxMemAllocSize << " bytes" << endl;

    // Path to the dataset folder and timestamp file
    string folderPath = "/home/odroid/Desktop/parallel/lidar/data/image_02/data/";
    string timestampFile = "/home/odroid/Desktop/parallel/lidar/data/image_02/timestamps.txt";

    // Load timestamps
    vector<string> timestamps = loadTimestamps(timestampFile);

    // Get all image files in the folder
    vector<string> imageFiles = getImageFiles(folderPath);
    if (imageFiles.empty()) {
        cerr << "No images found in the folder!" << endl;
        return -1;
    }

    // Charger MobileNet-SSD
    Net net = loadModel();

    // Variables for object tracking
    vector<int> objectIDs;
    vector<Rect> prevBboxes;
    vector<Point2f> prevCenters;
    int nextID = 0;

    // Output file to save coordinates and timestamps
    ofstream outputFile("object_trajectories.txt");
    if (!outputFile.is_open()) {
        cerr << "Could not open output file!" << endl;
        return -1;
    }

    // Process each image in the dataset
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        Mat img = imread(imageFiles[i], IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Impossible de charger l'image: " << imageFiles[i] << endl;
            continue;
        }

        // Detect objects and get bounding boxes and centers
        vector<Point2f> centers;
        vector<Rect> bboxes = detectObjects(net, img, centers);

        // Assign IDs to objects based on IoU with previous frame
        vector<int> currentIDs(bboxes.size(), -1);
        if (!prevBboxes.empty()) {
            vector<float> ious = computeIoU_GPU(context, queue, kernel, bboxes, prevBboxes);
            for (size_t j = 0; j < bboxes.size(); ++j) {
                for (size_t k = 0; k < prevBboxes.size(); ++k) {
                    if (ious[j * prevBboxes.size() + k] > 0.5) { // Threshold for matching objects
                        currentIDs[j] = objectIDs[k];
                        break;
                    }
                }
                if (currentIDs[j] == -1) { // New object
                    currentIDs[j] = nextID++;
                }
            }
        } else {
            for (size_t j = 0; j < bboxes.size(); ++j) {
                currentIDs[j] = nextID++;
            }
        }

        // Calculate velocities
        vector<Point2f> velocities(centers.size(), Point2f(0, 0));
        if (!prevCenters.empty()) {
            for (size_t j = 0; j < centers.size(); ++j) {
                if (currentIDs[j] != -1 && currentIDs[j] < prevCenters.size()) {
                    velocities[j] = centers[j] - prevCenters[currentIDs[j]];
                }
            }
        }

        // Save object coordinates, velocities, and timestamps to file
        for (size_t j = 0; j < bboxes.size(); ++j) {
            outputFile << "ID: " << currentIDs[j] << ", Center: (" << centers[j].x << ", " << centers[j].y
                       << "), Velocity: (" << velocities[j].x << ", " << velocities[j].y
                       << "), Timestamp: " << timestamps[i] << endl;
        }

        // Update previous bounding boxes, centers, and IDs
        prevBboxes = bboxes;
        prevCenters = centers;
        objectIDs = currentIDs;

        // Display the image with detected objects and centers
        imshow("Detected Objects", img);
        waitKey(1); // Wait for a short time to display the image
    }

    // Cleanup
    outputFile.close();
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    double endTime = (double)getTickCount();
    double elapsedTime = (endTime - startTime) / getTickFrequency();
    cout << "Total Processing Time: " << elapsedTime << " seconds" << endl;

    return 0;
}
