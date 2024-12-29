import mygputest.shuffle_copy as shuffle_copy
num_treads_per_block = 512
time_elapsed = []
time_elapsed_rnd=[]
for i in range(1,6):
    num_blocks = pow(10,i)
    numElements = num_blocks * num_treads_per_block
    print("***** num_blocks" + str(num_blocks) + "  no shared, consecutive *****")
    time_elapsed.append(shuffle_copy.test_copy_allkinds(num_treads_per_block, numElements, False, True))
    print("***** num_blocks" + str(num_blocks) + "  shared, consecutive *****")
    time_elapsed.append(shuffle_copy.test_copy_allkinds(num_treads_per_block, numElements, True, True))
    # print("***** no shared, not consecutive *****")
    # time_elapsed.append(shuffle_copy.test_copy_allkinds(num_treads_per_block, numElements, False, False))
    # print("***** shared, not consecutive *****")
    # time_elapsed.append(shuffle_copy.test_copy_allkinds(num_treads_per_block, numElements, True, False))
waithere=1



