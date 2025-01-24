//g++ -std=c++11 -fopenmp -o code1 code1.cpp `pkg-config --cflags --libs opencv4`


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <omp.h>  // Include OpenMP
using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"};

int main(int argc, char **argv)
{
    CV_TRACE_FUNCTION();

    // Paths to the model files
    String modelTxt = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/MobileNetSSD_deploy.prototxt";
    String modelBin = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/MobileNetSSD_deploy.caffemodel";
    String imageFile = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";

    // Load the network
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    net.setPreferableTarget(DNN_TARGET_CPU);  // Use CPU explicitly

    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }

    // Load the image
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    // Preprocess the image
    Mat img2;
    resize(img, img2, Size(300, 300));
    Mat inputBlob = blobFromImage(img2, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);

    // Set the input
    net.setInput(inputBlob, "data");

    // Measure time
    double time = (double)getTickCount();
    Mat detection = net.forward("detection_out");
    double time_elapsed = ((double)getTickCount() - time) / getTickFrequency();
    std::cout << "Time spent: " << time_elapsed << " seconds" << std::endl;

    // Set OpenMP to use 2 threads
    omp_set_num_threads(2);

    // Process the output
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    ostringstream ss;
    float confidenceThreshold = 0.2;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            int idx = static_cast<int>(detectionMat.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            Rect object(xLeftBottom, yLeftBottom,
                        xRightTop - xLeftBottom,
                        yRightTop - yLeftBottom);

            #pragma omp critical  // Ensure thread safety for console output and image manipulation
            {
                rectangle(img, object, Scalar(0, 255, 0), 2);
                cout << CLASSES[idx] << ": " << confidence << endl;

                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = CLASSES[idx] + ": " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                putText(img, label, Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            }
        }
    }

    // Display the output
    imshow("detections", img);
    waitKey();

    return 0;
}
