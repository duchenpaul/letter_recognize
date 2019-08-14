import shutil
import os
import config

def purge_folders(folder):
    '''Delete all files and dirs under dir: folder'''

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path, ignore_errors=True)
        except Exception as e:
            raise


def purge_models():
    for folder in [config.MODEL_DIR, config.MODEL_LOG]:
        print('Purge folder ' + folder)
        purge_folders(folder)


if __name__ == '__main__':
    purge_models()
