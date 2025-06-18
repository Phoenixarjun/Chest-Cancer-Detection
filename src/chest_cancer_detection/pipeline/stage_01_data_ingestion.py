from chest_cancer_detection.config.configuration import ConfigurationManager
from chest_cancer_detection.components.data_ingestion import DataIngestion
from chest_cancer_detection import logger
import zipfile

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        try:
            data_ingestion.extract_zip_file()
        except zipfile.BadZipFile:
            print("Downloaded file is not a valid zip file. Please check the download URL or file content.")
        except Exception as e:
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e