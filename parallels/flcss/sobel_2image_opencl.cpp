#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <chrono> // Pour mesurer le temps
#include <string> // Inclure cette bibliothèque pour utiliser 'std::string'

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "%s failed with error %d\n", msg, err); \
        exit(1); \
    }

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

// Fonction pour vérifier les erreurs de compilation OpenCL
void checkBuildLog(cl_program program, cl_device_id device) {
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

    char* buildLog = (char*)malloc(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
    fprintf(stderr, "Build log:\n%s\n", buildLog);
    free(buildLog);
}

int main() {
    // Utilisation de std::string pour les chemins des images
    std::string imageFile1 = "/home/odroid/Desktop/flcss/img/right/3.jpeg";
    std::string imageFile2 = "/home/odroid/Desktop/flcss/img/lift/3.jpeg";

    // Charger les images
    cv::Mat inputImage1 = cv::imread(imageFile1, cv::IMREAD_GRAYSCALE);
    cv::Mat inputImage2 = cv::imread(imageFile2, cv::IMREAD_GRAYSCALE);

    // Vérifier si les images ont été chargées correctement
    if (inputImage1.empty()) {
        fprintf(stderr, "Erreur de chargement de l'image : %s\n", imageFile1.c_str());
        return -1;
    }
    if (inputImage2.empty()) {
        fprintf(stderr, "Erreur de chargement de l'image : %s\n", imageFile2.c_str());
        return -1;
    }

    // Redimensionner les images
    int newWidth = 512;
    int newHeight = 512;
    cv::resize(inputImage1, inputImage1, cv::Size(newWidth, newHeight));
    cv::resize(inputImage2, inputImage2, cv::Size(newWidth, newHeight));

    int width = inputImage1.cols;
    int height = inputImage1.rows;
    size_t imageSize = width * height;

    // Créer les images de sortie
    cv::Mat outputImage1(height, width, CV_8UC1);
    cv::Mat outputImage2(height, width, CV_8UC1);

    cl_int err;

    // Obtenir la plateforme et le périphérique OpenCL
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "Getting platform");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "Getting device");

    // Créer le contexte OpenCL et la file de commandes
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Creating context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err, "Creating command queue");

    // Créer des buffers mémoire
    cl_mem inputBuffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &err);
    CHECK_ERROR(err, "Creating input buffer 1");

    cl_mem inputBuffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, &err);
    CHECK_ERROR(err, "Creating input buffer 2");

    cl_mem outputBuffer1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, NULL, &err);
    CHECK_ERROR(err, "Creating output buffer 1");

    cl_mem outputBuffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, NULL, &err);
    CHECK_ERROR(err, "Creating output buffer 2");

    // Copier les images sur le périphérique
    err = clEnqueueWriteBuffer(queue, inputBuffer1, CL_TRUE, 0, imageSize, inputImage1.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Writing input image 1 to device");

    err = clEnqueueWriteBuffer(queue, inputBuffer2, CL_TRUE, 0, imageSize, inputImage2.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Writing input image 2 to device");

    // Créer le programme OpenCL
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_ERROR(err, "Creating program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building program\n");
        checkBuildLog(program, device);
        exit(1);
    }

    // Créer le noyau OpenCL
    cl_kernel kernel = clCreateKernel(program, "sobel_filter", &err);
    CHECK_ERROR(err, "Creating kernel");

    // Définir les arguments du noyau
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer1);
    CHECK_ERROR(err, "Setting kernel arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer1);
    CHECK_ERROR(err, "Setting kernel arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(int), &width);
    CHECK_ERROR(err, "Setting kernel arg 2");

    err = clSetKernelArg(kernel, 3, sizeof(int), &height);
    CHECK_ERROR(err, "Setting kernel arg 3");

    // Exécuter le noyau
    size_t globalSize[2] = { (size_t)width, (size_t)height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "Enqueueing kernel");

    // Attendre la fin de l'exécution
    clFinish(queue);

    // Lire les résultats
    err = clEnqueueReadBuffer(queue, outputBuffer1, CL_TRUE, 0, imageSize, outputImage1.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Reading output buffer 1");

    // Appliquer le filtre Sobel à la deuxième image
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer2);
    CHECK_ERROR(err, "Setting kernel arg 0 for image 2");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer2);
    CHECK_ERROR(err, "Setting kernel arg 1 for image 2");

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "Enqueueing kernel for image 2");

    clFinish(queue);

    err = clEnqueueReadBuffer(queue, outputBuffer2, CL_TRUE, 0, imageSize, outputImage2.data, 0, NULL, NULL);
    CHECK_ERROR(err, "Reading output buffer 2");

    // Afficher les images résultantes
    cv::imshow("Sobel Output Image 1", outputImage1);
    cv::imshow("Sobel Output Image 2", outputImage2);
    cv::waitKey(0);

    // Nettoyer les ressources OpenCL
    clReleaseMemObject(inputBuffer1);
    clReleaseMemObject(inputBuffer2);
    clReleaseMemObject(outputBuffer1);
    clReleaseMemObject(outputBuffer2);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
