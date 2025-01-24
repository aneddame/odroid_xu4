#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath> // Pour std::sqrt
#include <algorithm> // Pour std::min et std::max
#include <chrono> // Pour mesurer le temps de traitement

int main() {
    const std::string imagePath = "/home/odroid/Desktop/flcss/img/right/3.jpeg";

    // Charger l'image en niveaux de gris
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Erreur : Impossible de charger l'image : " << imagePath << std::endl;
        return -1;
    }

    // Redimensionner l'image d'entrée
    int newWidth = 512;  // Largeur souhaitée
    int newHeight = 512; // Hauteur souhaitée
    cv::resize(inputImage, inputImage, cv::Size(newWidth, newHeight));

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Créer une image de sortie initialisée à zéro
    cv::Mat outputImage = cv::Mat::zeros(height, width, CV_8UC1);

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

    // Mesurer le temps de traitement
    auto startTime = std::chrono::high_resolution_clock::now();

    // Appliquer le filtre Sobel (manuellement)
    for (int y = 1; y < height - 1; ++y) {  // Éviter les bords
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;

            // Parcourir le voisinage 3x3
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel = inputImage.at<uchar>(y + ky, x + kx);
                    gx += pixel * sobelX[ky + 1][kx + 1];
                    gy += pixel * sobelY[ky + 1][kx + 1];
                }
            }

            // Calculer la magnitude du gradient
            int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));

            // Limiter la valeur entre 0 et 255
            magnitude = std::max(0, std::min(255, magnitude));

            outputImage.at<uchar>(y, x) = static_cast<uchar>(magnitude);
        }
    }

    // Mesurer le temps écoulé
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    // Afficher uniquement l'image filtrée
    cv::imshow("Filtre Sobel", outputImage);

    // Sauvegarder l'image de sortie
    if (!cv::imwrite("sobel_manual_output.jpg", outputImage)) {
        std::cerr << "Erreur : Impossible de sauvegarder l'image de sortie." << std::endl;
        return -1;
    }

    // Afficher le temps de traitement
    std::cout << "Temps de traitement : " << elapsed.count() << " secondes." << std::endl;

    // Attendre une touche avant de quitter
    cv::waitKey(0);

    return 0;
}
