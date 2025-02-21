#include "mnist_utils.h"

// Function to read IDX3-UBYTE Images files
std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);
    // std::cout<<static_cast<int>(numImagesBytes[0])<<"  "<<static_cast<int>(numImagesBytes[1])<<"  "<<
    //     (int)static_cast<unsigned char>(numImagesBytes[2])<<"  "<<static_cast<int>(numImagesBytes[3])<<"  "<<std::endl;

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | (static_cast<unsigned char>(numRowsBytes[1]) << 16) | (static_cast<unsigned char>(numRowsBytes[2]) << 8) | static_cast<unsigned char>(numRowsBytes[3]);
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | (static_cast<unsigned char>(numColsBytes[1]) << 16) | (static_cast<unsigned char>(numColsBytes[2]) << 8) | static_cast<unsigned char>(numColsBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(numRows * numCols);
        file.read((char*)(image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}


// Function to read IDX3-UBYTE Label files
std::vector<std::vector<unsigned char>> readLabelFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(1);
        file.read((char*)(image.data()), 1);

        images.push_back(image);
    }

    file.close();

    return images;
}

void readMNISTFiles(const std::string& imagesFilename, const std::string& labelsFilename,
                     std::vector<cv::Mat>& imagesData, std::vector<int>& labelsData) {

    std::vector<std::vector<unsigned char>> imagesFile = readIDX3UByteFile(imagesFilename);
    std::vector<std::vector<unsigned char>> labelsFile = readLabelFile(labelsFilename);

    for (int imgCnt = 0; imgCnt < static_cast<int>(imagesFile.size()); imgCnt++) {
        int rowCounter = 0;
        int colCounter = 0;

        cv::Mat tempImg = cv::Mat::zeros(cv::Size(28, 28), CV_8UC1);
        for (int i = 0; i < static_cast<int>(imagesFile[imgCnt].size()); i++) {
            tempImg.at<uchar>(cv::Point(colCounter++, rowCounter)) = static_cast<int>(imagesFile[imgCnt][i]);
            if ((i) % 28 == 0) {
                rowCounter++;
                colCounter = 0;
                if (i == 756) // when rowCounter reach 28, we will break the loop
                    break;
            }
        }
        imagesData.push_back(tempImg);
        labelsData.push_back(static_cast<int>(labelsFile[imgCnt][0]));
        // to visualize each image ,n dataset  to check only
        // cv::imshow("TempImg",tempImg);
        // int k = cv::waitKey(0);
    }

}