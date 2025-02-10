#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <dirent.h> // For directory handling

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Définition des classes MobileNet-SSD
string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

// Fonction pour calculer l'Intersection over Union (IoU)
float computeIoU(const Rect& bbox1, const Rect& bbox2) {
    int x1 = max(bbox1.x, bbox2.x);
    int y1 = max(bbox1.y, bbox2.y);
    int x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    int y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

    int intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    int bbox1Area = bbox1.width * bbox1.height;
    int bbox2Area = bbox2.width * bbox2.height;

    int unionArea = bbox1Area + bbox2Area - intersectionArea;

    return float(intersectionArea) / float(unionArea);
}

// Fonction pour charger les timestamps depuis un fichier
vector<string> loadTimestamps(const string& timestampFile) {
    vector<string> timestamps;
    ifstream file(timestampFile);
    if (!file.is_open()) {
        cerr << "Impossible d'ouvrir le fichier des timestamps: " << timestampFile << endl;
        exit(-1);
    }
    string line;
    while (getline(file, line)) {
        timestamps.push_back(line);
    }
    file.close();
    return timestamps;
}

// Fonction pour récupérer les fichiers images dans un dossier
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
        cerr << "Impossible d'ouvrir le dossier: " << folderPath << endl;
        exit(-1);
    }
    // Trier les fichiers pour assurer l'ordre
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

// Détection d'objets
vector<Rect> detectObjects(Net &net, Mat &img, vector<Point2f>& centers) {
    Mat img2;
    resize(img, img2, Size(300, 300));  // Redimensionner pour améliorer les performances

    Mat inputBlob = blobFromImage(img2, 0.007843, Size(224, 224), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inputBlob, "data");

    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    float confidenceThreshold = 0.2;
    vector<Rect> bboxes;

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

            bboxes.push_back(bbox);
            centers.push_back(center);
            rectangle(img, bbox, Scalar(0, 255, 0), 2);
            putText(img, CLASSES[idx] + " " + to_string(confidence), Point(xLeftBottom, yLeftBottom - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            circle(img, center, 3, Scalar(255, 0, 0), -1); // Dessiner le point central
        }
    }
    return bboxes;
}

int main() {
    double startTime = (double)getTickCount();

    // Chemin du dossier et fichier des timestamps
    string folderPath = "/home/odroid/Desktop/parallel/lidar/data/image_03/data/";
    string timestampFile = "/home/odroid/Desktop/parallel/lidar/data/image_03/timestamps.txt";

    // Charger les timestamps
    vector<string> timestamps = loadTimestamps(timestampFile);

    // Obtenir les images du dossier
    vector<string> imageFiles = getImageFiles(folderPath);
    if (imageFiles.empty()) {
        cerr << "Aucune image trouvée dans le dossier !" << endl;
        return -1;
    }

    // Charger MobileNet-SSD
    Net net = loadModel();

    // Variables pour le suivi des objets
    vector<int> objectIDs;
    vector<Rect> prevBboxes;
    int nextID = 0;

    // Fichier de sortie pour sauvegarder les coordonnées et timestamps
    ofstream outputFile("object_trajectories2.txt");
    if (!outputFile.is_open()) {
        cerr << "Impossible d'ouvrir le fichier de sortie !" << endl;
        return -1;
    }

    // Traitement de chaque image
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        Mat img = imread(imageFiles[i], IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Impossible de charger l'image: " << imageFiles[i] << endl;
            continue;
        }

        // Détecter les objets et récupérer les boîtes englobantes et centres
        vector<Point2f> centers;
        vector<Rect> bboxes = detectObjects(net, img, centers);

        // Assigner des IDs aux objets détectés
        vector<int> currentIDs(bboxes.size(), -1);
        for (size_t j = 0; j < bboxes.size(); ++j) {
            for (size_t k = 0; k < prevBboxes.size(); ++k) {
                float iou = computeIoU(bboxes[j], prevBboxes[k]);
                if (iou > 0.5) {
                    currentIDs[j] = objectIDs[k];
                    break;
                }
            }
            if (currentIDs[j] == -1) {
                currentIDs[j] = nextID++;
            }
        }

        // Sauvegarde des données
        for (size_t j = 0; j < bboxes.size(); ++j) {
            outputFile << "ID: " << currentIDs[j] << ", Center: (" << centers[j].x << ", " << centers[j].y
                       << "), Timestamp: " << timestamps[i] << endl;
        }

        prevBboxes = bboxes;
        objectIDs = currentIDs;

        imshow("Detected Objects", img);
        waitKey(1);
    }

    outputFile.close();
    double endTime = (double)getTickCount();
    cout << "Temps total de traitement: " << (endTime - startTime) / getTickFrequency() << " secondes" << endl;

    return 0;
}
