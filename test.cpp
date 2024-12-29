#include "extern.h"
#include <iostream>

int main() {
    int blockSize = 256;
    int numElements = blockSize * 1000;
    float time_elapsed = test_copy_allkinds(blockSize, numElements, false, true);

    if(time_elapsed < 0) {
        std::cout << "Error in test_just_copy" << std::endl;
    } else {
        std::cout << "Time for just copy: " << time_elapsed << " ms" << std::endl;
    }

    return 0;
}