import toolkit_file
import config

def purge_models():
    toolkit_file.purge_folder(config.TRAINING_DATA_DIR)
    toolkit_file.purge_folder(config.MODEL_DIR)

if __name__ == '__main__':
    pass
