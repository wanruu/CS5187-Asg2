import os


test_root = "StereoMatchingTestings"
test_dirs = os.listdir(test_root)
test_paths = {dir.lower(): os.path.join(test_root, dir) for dir in test_dirs}


result_path = "results/"