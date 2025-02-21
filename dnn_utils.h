#ifndef DNN_UTILS_H
#define DNN_UTILS_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

cv::Ptr<cv::ml::ANN_MLP> trainingModel(const std::vector<cv::Mat>& imagesData, const std::vector<int>& labelsData);
float testModelAccuracy(const cv::Ptr<cv::ml::ANN_MLP>& mlp, const  std::vector<cv::Mat>& imagesTestData, std::vector<int>& labelsTestData);

#endif
