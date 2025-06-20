from chest_cancer_detection import logger
from chest_cancer_detection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from chest_cancer_detection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from chest_cancer_detection.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from chest_cancer_detection.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model Stage"

try:
        logger.info(f"******************************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Trainer Stage"


try:
    logger.info("*******************************")
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


import dagshub
dagshub.init(repo_owner='phoenixarjun007', repo_name='Chest-Cancer-Detection', mlflow=True)


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info("*******************************")
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e