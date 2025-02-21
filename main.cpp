#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "mnist_utils.h"
#include "dnn_utils.h"


int main() {

    std::string filename = "mnist_data/train-images.idx3-ubyte";
    std::string label_filename = "mnist_data/train-labels.idx1-ubyte";

    std::vector<cv::Mat> imagesData;  // Store your images
    std::vector<int> labelsData;      // Corresponding labels

    // Void function will store the data to imagesData and labelsData
    readMNISTFiles(filename, label_filename, imagesData, labelsData);


    // 2nd Step: Model Training (uncomment two lines below if you want to train your own model)
    // cv::Ptr<cv::ml::ANN_MLP> mlp = trainingModel(imagesData, labelsData);
    // mlp->save("trained_mnist_model_2HU_100.xml");


    // 3rd Step: Load the Trained model
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::load("trained_mnist_model_2HU_100.xml");
    std::cout<< "Successfully load the model" <<std::endl;

    // Load the Test Image
    std::string test_img_filename = "mnist_data/t10k-images.idx3-ubyte";
    std::string test_label_filename = "mnist_data/t10k-labels.idx1-ubyte";
  
    std::vector<cv::Mat> imagesTestData;  // Store your test images
    std::vector<int> labelsTestData;      // Corresponding labels

    readMNISTFiles(test_img_filename, test_label_filename, imagesTestData, labelsTestData);

    // Model Accuracy for Training Data
    float train_acc = testModelAccuracy(mlp, imagesData, labelsData);
    std::cout<< "Prediction Accuracy of Training Data: "<< train_acc <<"%" <<std::endl;

    // Model Accuracy for Test Data
    float test_acc = testModelAccuracy(mlp, imagesTestData, labelsTestData);
    std::cout<< "Prediction Accuracy of Test Data: "<< test_acc<<"%" <<std::endl;
    
    std::cout<<std::endl;
    system("pause");

    return 0;
}
