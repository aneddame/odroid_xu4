//g++ -o object_detection_fuzzy_lcss main.cpp `pkg-config --cflags --libs opencv4` -fopenmp -std=c++11

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// Fonction d'appartenance floue
float fuzzy_membership(float distance, float c = 0.2, float d = 1.6) {
    if (distance <= c) return 1.0;
    else if (distance <= d) return (d - distance) / (d - c);
    else return 0.0;
}

// Implémentation de F_LCSS
float fuzzy_lcss(const vector<Point2f>& traj1, const vector<Point2f>& traj2, float c = 0.2, float d = 1.6) {
    int n = traj1.size(), m = traj2.size();
    vector<vector<float>> dp(n + 1, vector<float>(m + 1, 0.0));

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float dist = norm(traj1[i - 1] - traj2[j - 1]);
            float membership = fuzzy_membership(dist, c, d);
            dp[i][j] = max({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + membership});
        }
    }

    return dp[n][m];
}

int main() {
    // Charger les images
    string imageFile1 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";
    string imageFile2 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";

    Mat img1 = imread(imageFile1, IMREAD_COLOR);
    Mat img2 = imread(imageFile2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        cerr << "Impossible de charger les images." << endl;
        return -1;
    }

    // Détection d'objets dans img1 avec OpenCV DNN
    String modelTxt = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/MobileNetSSD_deploy.prototxt";
    String modelBin = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/MobileNetSSD_deploy.caffemodel";

    // Utilisation explicite du namespace cv::dnn
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
    if (net.empty()) {
        cerr << "Impossible de charger le modèle DNN." << endl;
        return -1;
    }

    // Utilisation explicite du namespace cv::dnn pour blobFromImage
    Mat inputBlob = cv::dnn::blobFromImage(img1, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(inputBlob, "data");

    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    bool objectDetected = false;
    float confidenceThreshold = 0.2;
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold) {
            objectDetected = true;
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img1.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img1.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img1.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img1.rows);

            Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
            rectangle(img1, object, Scalar(0, 255, 0), 2);

            cout << "Objet détecté avec confiance " << confidence << endl;
        }
    }

    imshow("Détections dans img1", img1);

    // Si un objet est détecté, appliquer F_LCSS sur img1 et img2
    if (objectDetected) {
        cout << "Objet détecté, application de F_LCSS sur img1 et img2." << endl;

        // Détection et description des points clés avec ORB
        Ptr<ORB> orb = ORB::create();
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;

        orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

        if (keypoints1.empty() || keypoints2.empty()) {
            cerr << "Aucun point clé détecté dans l'une des images." << endl;
            return -1;
        }

        // Matcher les descripteurs avec BFMatcher
        BFMatcher matcher(NORM_HAMMING);
        vector<DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        // Filtrer les bons matches
        sort(matches.begin(), matches.end(), [](const DMatch& a, const DMatch& b) {
            return a.distance < b.distance;
        });
        matches.resize(min(50, (int)matches.size())); // Garde les 50 meilleurs matches

        // Extraire les trajectoires des points correspondants
        vector<Point2f> traj1, traj2;
        for (const auto& match : matches) {
            traj1.push_back(keypoints1[match.queryIdx].pt);
            traj2.push_back(keypoints2[match.trainIdx].pt);
        }

        // Appliquer F_LCSS sur les trajectoires
        float similarity = fuzzy_lcss(traj1, traj2);
        cout << "Similarité des trajectoires (F_LCSS) : " << similarity << endl;

        // Afficher les correspondances
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
        imshow("Correspondances", img_matches);
    } else {
        cout << "Aucun objet détecté dans img1, F_LCSS non appliqué." << endl;
    }

    waitKey(0);
    return 0;
}
