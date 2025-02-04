#ifdef CVLIB_EXPORTS
#define CVLIB_API __declspec(dllexport)
#else
#define CVLIB_API __declspec(dllimport)
#endif

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

// Declare your API functions with the appropriate export/import specifier
#ifdef __cplusplus
extern "C" {
#endif

	CVLIB_API int MoveImageToDeviceCv(cv::Mat);

#ifdef __cplusplus
}
#endif