
//compile  g++ -std=c++11 -o yolov4 yolov4.cpp `pkg-config --cflags --libs opencv4`
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono> // For timing

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
    // Load YOLOv4 model
    String modelConfiguration = "/home/odroid/darknet/cfg/yolov4.cfg"; // Path to YOLOv4 config file
    String modelWeights = "/home/odroid/darknet/cfg/yolov4.weights"; // Path to YOLOv4 weights file
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    // Set backend and target
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Load the image
    Mat image = imread("/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg"); // Replace with your image path
    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Start timing
    auto start = chrono::high_resolution_clock::now();

    // Get the blob from the image
    Mat blob;
    blobFromImage(image, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);

    // Set the input for the network
    net.setInput(blob);

    // Get the output layer names
    vector<String> layerNames = net.getLayerNames();
    vector<int> outLayers = net.getUnconnectedOutLayers();
    vector<String> outLayerNames;
    for (int i = 0; i < outLayers.size(); ++i) {
        outLayerNames.push_back(layerNames[outLayers[i] - 1]);
    }

    // Run forward pass to get output
    vector<Mat> outs;
    net.forward(outs, outLayerNames);

    // Process the output
    float confThreshold = 0.5;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        Mat& detectionMat = outs[i];
        for (int j = 0; j < detectionMat.rows; ++j) {
            Mat row = detectionMat.row(j);
            // Get confidence score
            Mat scores = row.colRange(5, detectionMat.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                // Get bounding box
                int centerX = (int)(row.at<float>(0) * image.cols);
                int centerY = (int)(row.at<float>(1) * image.rows);
                int width = (int)(row.at<float>(2) * image.cols);
                int height = (int)(row.at<float>(3) * image.rows);
                int x = centerX - width / 2;
                int y = centerY - height / 2;

                boxes.push_back(Rect(x, y, width, height));
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
            }
        }
    }

    // Perform non-maxima suppression to eliminate redundant overlapping boxes
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, 0.4, indices);

    // Draw the bounding boxes on the image
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        rectangle(image, box, Scalar(0, 255, 0), 2);
    }

    // End timing
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // Display the time spent
    cout << "Time spent: " << elapsed.count() << " seconds" << endl;

    // Display the image
    namedWindow("Detections", WINDOW_NORMAL);
    imshow("Detections", image);
    waitKey(0);

    return 0;
}
