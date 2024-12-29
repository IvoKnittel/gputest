import os
import sys

# Print the current sys.path
print("Current sys.path:", sys.path)

# Specify the directory where the DLLs are located
dll_directory = 'C:\\path\\to\\dlls'

# Add the directory to the PATH environment variable
os.environ['PATH'] = dll_directory + os.pathsep + os.environ['PATH']

# Print the updated PATH
print("Updated PATH:", os.environ['PATH'])

# Try to import the module
try:
    import mygputest.shuffle_copy as shuffle_copy
    print("Module imported successfully")

    # Call functions to verify import
    result_random_index = shuffle_copy.test_random_index_copy()
    result_just_copy = shuffle_copy.test_just_copy()

    print("test_random_index_copy result:", result_random_index)
    print("test_just_copy result:", result_just_copy)
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



