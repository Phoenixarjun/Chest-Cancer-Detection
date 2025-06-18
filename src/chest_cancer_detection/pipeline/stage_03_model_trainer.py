from chest_cancer_detection.config.configuration import ConfigurationManager
from chest_cancer_detection.components.model_trainer import Training
from chest_cancer_detection import logger
import zipfile

STAGE_NAME = "Model Trainer Stage"

class ModelTrainingPipeline:

  def __init__(self):
    pass

  def main(self):
    config = ConfigurationManager()
    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.train()

if __name__ == "__main__":
  try:
    logger.info("*******************************")
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<")
  except Exception as e:
    logger.exception(e)
    raise e