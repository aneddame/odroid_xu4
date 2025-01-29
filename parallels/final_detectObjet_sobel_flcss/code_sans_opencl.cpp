#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Définition des classes MobileNet-SSD
string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

// Noyaux Sobel
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

// Détection d'objets
Mat detectObjects(Net &net, Mat &img) {
    Mat img2;
    resize(img, img2, Size(300, 300));

    Mat inputBlob = blobFromImage(img2, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inputBlob, "data");

    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    float confidenceThreshold = 0.2;
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold) {
            int idx = static_cast<int>(detectionMat.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            rectangle(img, Rect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom), Scalar(0, 255, 0), 2);
            putText(img, CLASSES[idx] + " " + to_string(confidence), Point(xLeftBottom, yLeftBottom - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        }
    }
    return img;
}

// Filtre Sobel manuel
void apply_sobel_manual(const Mat& input, Mat& gradientX, Mat& gradientY, Mat& magnitude) {
    CV_Assert(input.type() == CV_8U);  // S'assurer que l'image est en niveaux de gris

    int width = input.cols;
    int height = input.rows;

    gradientX = Mat::zeros(input.size(), CV_32F);
    gradientY = Mat::zeros(input.size(), CV_32F);
    magnitude = Mat::zeros(input.size(), CV_32F);

    for (int y = 1; y < height - 1; ++y) {  // Éviter les bords
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0.0f, gy = 0.0f;

            // Parcourir le voisinage 3x3
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel = input.at<uchar>(y + ky, x + kx);
                    gx += pixel * sobelX[ky + 1][kx + 1];
                    gy += pixel * sobelY[ky + 1][kx + 1];
                }
            }

            gradientX.at<float>(y, x) = gx;
            gradientY.at<float>(y, x) = gy;

            // Magnitude des gradients
            magnitude.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }
    }

    // Normaliser pour affichage
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);
    magnitude.convertTo(magnitude, CV_8U);
}

// Calculer la similarité F_LCSS
float F_LCSS(const Mat &sobel1, const Mat &sobel2) {
    int width = sobel1.cols;
    int height = sobel1.rows;
    float similarity = 0.0f;
    int totalPixels = width * height;

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
    string imageFile1 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";
    string imageFile2 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";

    Mat img1 = imread(imageFile1, IMREAD_COLOR);
    Mat img2 = imread(imageFile2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        cerr << "Impossible de charger les images !" << endl;
        return -1;
    }

    resize(img1, img1, Size(300, 300));
    resize(img2, img2, Size(300, 300));

    // Charger MobileNet-SSD et détecter les objets
    Net net = loadModel();
    Mat img1Detected = detectObjects(net, img1);
    Mat img2Detected = detectObjects(net, img2);

    
    // Convertir en niveaux de gris
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Appliquer Sobel manuellement
    Mat gradX1, gradY1, sobel1;
    Mat gradX2, gradY2, sobel2;

    apply_sobel_manual(gray1, gradX1, gradY1, sobel1);
    apply_sobel_manual(gray2, gradX2, gradY2, sobel2);

    // Calculer la similarité F_LCSS
    float similarity = F_LCSS(sobel1, sobel2);
    float similarityPercentage = similarity * 100;
    cout << "Fuzzy Similarity (F_LCSS) = " << similarity << endl;
    cout << "Similarity Percentage = " << similarityPercentage << "%" << endl;

    // Correspondances de points d'intérêt
    vector<Point2f> traj1, traj2;
    goodFeaturesToTrack(gray1, traj1, 100, 0.01, 10);
    goodFeaturesToTrack(gray2, traj2, 100, 0.01, 10);

    // Afficher les points détectés
    Mat img1Copy = img1Detected.clone();
    Mat img2Copy = img2Detected.clone();
    for (size_t i = 0; i < traj1.size() && i < traj2.size(); ++i) {
        circle(img1Copy, traj1[i], 5, Scalar(0, 0, 255), -1);
        circle(img2Copy, traj2[i], 5, Scalar(255, 0, 0), -1);
    }

    // Combiner les images
    Mat img_matches;
    hconcat(img1Copy, img2Copy, img_matches);

    // Calculer le temps de traitement
    double endTime = (double)getTickCount();
    double processingTime = (endTime - startTime) / getTickFrequency();
    cout << "Temps de traitement : " << processingTime << " secondes" << endl;

   
    // Afficher les résultats
    imshow("Détection d'objets", img_matches);
    waitKey(0);

    return 0;
}
