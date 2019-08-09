import shutil
import config

def purge_models():
    for folder in [config.TRAINING_DATA_DIR, config.MODEL_DIR]:
        shutil.rmtree(folder, ignore_errors=True)

if __name__ == '__main__':
    pass
