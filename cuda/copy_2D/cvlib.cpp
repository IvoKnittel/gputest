#include <iostream>
#include <stdexcept>
#include "extern.h"
#include "cvlib.h"


EXTERN_C  int MoveImageToDeviceCv(cv::Mat image8) {

    // Check if the image is grayscale
    if (image8.channels() != 1) {
		return -1;
	}

    int width = image8.cols;
    int height = image8.rows;

    // Check if the image size is a multiple of 16
    if (height % _blocksize != 0 || width % _blocksize != 0) {
        return -1;
    }

    if (!is_enabled) {
        enableDevice();
    }

    if (!image_init_done) {
        // Calculate the new dimensions that are the closest multiples of 16
        image_num_blocks[vert_dir] = width / _blocksize;
        image_num_blocks[horiz_dir] = height / _blocksize;
        image_num_pixels[vert_dir] = image_num_blocks[vert_dir] * _blocksize;
        image_num_pixels[horiz_dir] = image_num_blocks[horiz_dir] * _blocksize;
        initialize(image_num_blocks);
    }
    MoveImageToDevice(image8.data);
    return 0;
}