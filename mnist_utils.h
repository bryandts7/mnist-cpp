#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename);
std::vector<std::vector<unsigned char>> readLabelFile(const std::string& filename);
void readMNISTFiles(const std::string& imagesFilename, const std::string& labelsFilename,
                     std::vector<cv::Mat>& imagesData, std::vector<int>& labelsData);

#endif
