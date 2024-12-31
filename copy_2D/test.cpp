#include "extern.h"
#include <iostream>

int main() {

    int blockSize = 256;
    int numElements = blockSize * 1000;

	inititalize(false, numElements);
    float time_elapsed = test_copy_gpu(blockSize, numElements, false);
	clear_all();

    if(time_elapsed < 0) {
        std::cout << "Error in test_just_copy" << std::endl;
    } else {
        std::cout << "Time for just copy: " << time_elapsed << " ms" << std::endl;
    }

    return 0;
}