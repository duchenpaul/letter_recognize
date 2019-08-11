import os
import toolkit_file

FONT_DIR = 'fonts'
TRAINING_DATA_DIR = 'training_data'
TEST_DATA_DIR = 'test_data'

MODEL_DIR = 'models'
MODEL_LOG = 'log'

char_set = list(range(ord('0'), ord('9') + 1)) + list(range(ord('A'), ord('Z') + 1)) + list(range(ord('a'), ord('z') + 1))
# char_set = list(range(ord('0'), ord('9') + 1))

for folder in [FONT_DIR, TRAINING_DATA_DIR, TEST_DATA_DIR, MODEL_DIR]:
    toolkit_file.create_folder(folder)

for x in char_set:
    toolkit_file.create_folder(os.path.join(TRAINING_DATA_DIR, str(x)))


IMG_SIZE = 28


if __name__ == '__main__':
    # import shutil
    # shutil.rmtree(TRAINING_DATA_DIR, ignore_errors=True)
    # toolkit_file.purge_folder(TRAINING_DATA_DIR)
    pass
