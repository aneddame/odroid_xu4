#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

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

    return dp[n][m] / min(n, m) * 100.0f;  // Résultat en pourcentage
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

// Fonction principale
int main() {
    // Start timer
    double startTime = (double)getTickCount();

    // Charger les images
    string imageFile1 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";
    string imageFile2 = "/home/odroid/Desktop/MobileNet-SSD/mobilenet/images/000456.jpg";

    Mat img1 = imread(imageFile1, IMREAD_COLOR);
    Mat img2 = imread(imageFile2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        cerr << "Impossible de charger les images." << endl;
        return -1;
    }

    // Stop timer after loading images
    double loadTime = ((double)getTickCount() - startTime) / getTickFrequency();
    cout << "Temps de chargement des images: " << loadTime << " secondes." << endl;

    // Redimensionner les images à 300x300
    resize(img1, img1, Size(300, 300));
    resize(img2, img2, Size(300, 300));

    // Convertir en niveaux de gris
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Appliquer Sobel manuellement
    Mat gradX1, gradY1, mag1;
    Mat gradX2, gradY2, mag2;

    apply_sobel_manual(gray1, gradX1, gradY1, mag1);
    apply_sobel_manual(gray2, gradX2, gradY2, mag2);

    // Stop timer after Sobel edge detection
    double sobelTime = ((double)getTickCount() - startTime) / getTickFrequency();
    cout << "Temps de traitement Sobel: " << sobelTime << " secondes." << endl;

    // Extraire les points forts des bords
    vector<Point2f> traj1, traj2;
    for (int y = 1; y < mag1.rows - 1; ++y) {
        for (int x = 1; x < mag1.cols - 1; ++x) {
            if (mag1.at<uchar>(y, x) > 50) {  // Seuillage pour extraire les bords
                traj1.emplace_back(x, y);
            }
            if (mag2.at<uchar>(y, x) > 50) {
                traj2.emplace_back(x, y);
            }
        }
    }

    // Appliquer F_LCSS sur les trajectoires extraites des bords
    float similarity = fuzzy_lcss(traj1, traj2);
    cout << "Similarité des trajectoires (F_LCSS) : " << similarity << "%" << endl;

    // Stop timer after F_LCSS
    double flcssTime = ((double)getTickCount() - startTime) / getTickFrequency();
    cout << "Temps de traitement F_LCSS: " << flcssTime << " secondes." << endl;

    // Afficher les correspondances des points détectés sur les images originales
    Mat img1Copy = img1.clone();
    Mat img2Copy = img2.clone();
    for (size_t i = 0; i < traj1.size() && i < traj2.size(); ++i) {
        circle(img1Copy, traj1[i], 2, Scalar(0, 0, 255), -1);  // Points rouges sur img1
        circle(img2Copy, traj2[i], 2, Scalar(255, 0, 0), -1);  // Points bleus sur img2
    }

    // Combiner les deux images originales côte à côte
    Mat img_matches;
    hconcat(img1Copy, img2Copy, img_matches);

    imshow("Correspondances des bords détectés", img_matches);
    waitKey(0);

    // Final time
    double totalTime = ((double)getTickCount() - startTime) / getTickFrequency();
    cout << "Temps total de traitement: " << totalTime << " secondes." << endl;

    return 0;
}
