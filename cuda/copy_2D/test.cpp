#include "extern.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>

int main() {

    // Set the logging level to suppress INFO messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // Read an image
	std::string image_path = "dut1.bmp";
    cv::Mat  image8 = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    int width = image8.cols;
    int height = image8.rows;
    std::cout << "Image width" << width << " Image height" << height << std::endl;

    // Calculate the new dimensions that are the closest multiples of 16
    image_num_blocks[vert_dir]  = width / _blocksize;
    image_num_blocks[horiz_dir] = height / _blocksize;
    image_num_pixels[vert_dir]  = image_num_blocks[vert_dir ] * _blocksize;
    image_num_pixels[horiz_dir] = image_num_blocks[horiz_dir] * _blocksize;

    // Crop the image to the new dimensions
    cv::Rect cropRegion(0, 0, image_num_pixels[vert_dir], image_num_pixels[horiz_dir]);
    cv::Mat croppedImage8 = image8(cropRegion);

    enableDevice();
    initialize(image_num_blocks);
    MoveImageToDevice(croppedImage8.data);
    image_copy();

	cv::Mat image_copy = cv::Mat(image_num_pixels[horiz_dir], image_num_pixels[vert_dir], CV_8UC1);
    
    width = image_copy.cols;
    height = image_copy.rows;
    std::cout << "Image Copy width" << width << " Image Copy height" << height << std::endl;

    GetImageFromDevice(image_copy.data);
    clear_all();
    
    // Create a window
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

    // Show the image inside the created window
    cv::imshow("Display window", image8);

    // Wait for any keystroke in the window
    cv::waitKey(0);

    // Create a window
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

    // Show the image inside the created window
    cv::imshow("Display window", image_copy);

    // Wait for any keystroke in the window
    cv::waitKey(0);


    /*int blockSize = 256;
    int numElements = blockSize * 1000;

	initialize(numElements);
    float time_elapsed = test_copy_gpu(blockSize, numElements, false);
	clear_all();

    if(time_elapsed < 0) {
        std::cout << "Error in test_just_copy" << std::endl;
    } else {
        std::cout << "Time for just copy: " << time_elapsed << " ms" << std::endl;
    }*/

    return 0;
}