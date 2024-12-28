//code extrait les trajectoires des points d'intérêt et applique F_LCSS pour évaluer leur similarité.
//g++ -std=c++11 -o fuzzy_lcss fuzzy_lcss.cpp `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
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

int main(int argc, char** argv) {
    // Charger deux images
    string imageFile1 = "/home/odroid/Desktop/download.jpeg";
    string imageFile2 = "/home/odroid/Desktop/idownload.jpeg";

    Mat img1 = imread(imageFile1, IMREAD_GRAYSCALE);
    Mat img2 = imread(imageFile2, IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cerr << "Impossible de charger les images." << endl;
        return -1;
    }

    // Détection et description des points clés avec ORB
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    if (keypoints1.empty() || keypoints2.empty()) {
        cerr << "Aucun point clé détecté." << endl;
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
    waitKey(0);

    return 0;
}
