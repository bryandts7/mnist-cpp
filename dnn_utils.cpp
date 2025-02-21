#include "dnn_utils.h"

cv::Ptr<cv::ml::ANN_MLP> trainingModel(const std::vector<cv::Mat>& imagesData, const std::vector<int>& labelsData){
    
    // 1st Step: Deep Neural Networks Model Architecture
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();

    int inputLayerSize = imagesData[0].total();
    int hiddenLayerSize = 100;
    int outputLayerSize = 10;

    cv::Mat layers = (cv::Mat_<int>(3,1) << inputLayerSize, hiddenLayerSize, outputLayerSize);
    mlp->setLayerSizes(layers);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0, 0);

    // 2nd Step: Preparing the Training Data
    int numSamples = imagesData.size();

    cv::Mat trainingData(numSamples, inputLayerSize, CV_32F);
    cv::Mat labelData(numSamples, outputLayerSize, CV_32F);

    for(int i=0; i<numSamples; i++){
        cv::Mat image = imagesData[i].reshape(1, 1);
        image.convertTo(trainingData.row(i), CV_32F);

        cv::Mat label = cv::Mat::zeros(1, outputLayerSize, CV_32F);
        label.at<float>(0, labelsData[i]) = 1.0;
        label.copyTo(labelData.row(i));

    }


    // 3rd Step: Training the Model
    cv::TermCriteria termCrit(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 500, 0.001);

    mlp->setTermCriteria(termCrit);

    mlp->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP, 0.0001);

    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, labelData);
    mlp->train(trainData);

    return mlp;
}

float testModelAccuracy(const cv::Ptr<cv::ml::ANN_MLP>& mlp, const  std::vector<cv::Mat>& imagesTestData, std::vector<int>& labelsTestData){
    float true_prediction = 0;
    // 4th Step: Predict the Test Images
    for (int i=0; i<imagesTestData.size(); i++){

        cv::Mat flattenedImage = imagesTestData[i].reshape(1, 1);
        cv::Mat input;
        flattenedImage.convertTo(input, CV_32F);

        cv::Mat output;
        mlp->predict(input, output);

        // Find the class with the highest probability
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);

        int predictedClass = classIdPoint.x;

        // To display test image and prediction
        std::cout << "Predicted class: " << predictedClass << " with confidence: " << confidence << std::endl;
        cv::imshow("TempImg", imagesTestData[i]);
        int k = cv::waitKey(0);

        if (predictedClass == labelsTestData[i]){
            true_prediction += 1;
        }

    }

    float accuracy = true_prediction * 100 / (float) imagesTestData.size();
    return accuracy;

}