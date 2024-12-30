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

	SHUFFLECOPY_API void inititalize(bool random_data, int numElements);
	SHUFFLECOPY_API void clear_all();

	SHUFFLECOPY_API float test_copy_gpu(const int blockSize, const int numElements, const bool use_shared);
	SHUFFLECOPY_API float test_random_index_copy(const int blockSize, const int numElements);

	SHUFFLECOPY_API float test_random_copy_cpu(const int numElements);
	SHUFFLECOPY_API float test_copy_cpu(const int numElements);

#ifdef __cplusplus
}
#endif