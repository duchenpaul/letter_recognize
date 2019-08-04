import os
import toolkit_file

FONT_DIR = 'fonts'
TRAINING_DATA_DIR = 'training_data'
TEST_DATA_DIR = 'test_data'

char_set = list(range(ord('0'), ord('9')+1)) + list(range(ord('A'), ord('Z')+1)) + list(range(ord('a'), ord('z')+1))

for folder in [FONT_DIR, TRAINING_DATA_DIR, TEST_DATA_DIR]:
    toolkit_file.create_folder(folder)

for x in char_set:
    toolkit_file.create_folder(os.path.join(TRAINING_DATA_DIR, str(x)))