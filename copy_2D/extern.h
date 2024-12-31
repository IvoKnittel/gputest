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

#include "gpu2d.h"

// Declare your API functions with the appropriate export/import specifier
#ifdef __cplusplus
extern "C" {
#endif

	IMAGECOPY_API void enableDevice() 
	IMAGECOPY_API void inititalize(int numBlocks[NUM_DIMENSIONS]);
	IMAGECOPY_API void MoveImageToDevice(Pixel* data)
	IMAGECOPY_API void image_copy()
	IMAGECOPY_API void GetImageFromDevice(Pixel* data, int stream_id);
	IMAGECOPY_API void clear_all();
	
#ifdef __cplusplus
}
#endif