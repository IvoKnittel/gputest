#include "cvlib.h"

EXTERN_C  int image_num_pixels(cv::Mat image8) {

    // Check if the image is grayscale
    if (image8.channels() != 1) {
		return -1;
	}

    int width = image8.cols;
    int height = image8.rows;

    if (!image_init_done) {
        image_num_pixels[vert_dir] = width;
        image_num_pixels[horiz_dir] = height;
		image_init_done=true;
    }
    return width*height;
}