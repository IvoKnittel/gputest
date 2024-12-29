#ifdef SHUFFLECOPY_EXPORTS
#define SHUFFLECOPY_API __declspec(dllexport)
#else
#define SHUFFLECOPY_API __declspec(dllimport)
#endif

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif


// Declare your API functions with the appropriate export/import specifier
#ifdef __cplusplus
extern "C" {
#endif

	SHUFFLECOPY_API float test_copy_allkinds(const int blockSize, const int numElements, const bool use_shared, const bool is_consecutive);

#ifdef __cplusplus
}
#endif