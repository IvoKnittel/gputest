#ifdef IMAGECOPY_EXPORTS
#define IMAGECOPY_API __declspec(dllexport)
#else
#define IMAGECOPY_API __declspec(dllimport)
#endif

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#include "kernel.h"
int  image_num_pixels[NUM_DIMENSIONS];
int  image_num_blocks[NUM_DIMENSIONS];
using namespace std;
const int horiz_dir = 0;
const int vert_dir = 1;

// Declare your API functions with the appropriate export/import specifier
#ifdef __cplusplus
extern "C" {
#endif

	IMAGECOPY_API void enableDevice();
	IMAGECOPY_API void initialize(int numBlocks[NUM_DIMENSIONS]);
	IMAGECOPY_API void MoveImageToDevice(Pixel* data);
	IMAGECOPY_API void image_copy();
	IMAGECOPY_API void GetImageFromDevice(Pixel* data);
	IMAGECOPY_API void clear_all();
	
#ifdef __cplusplus
}
#endif