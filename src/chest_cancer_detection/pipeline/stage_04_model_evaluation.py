from chest_cancer_detection.config.configuration import ConfigurationManager
from chest_cancer_detection.components.model_evaluation_mlflow import Evaluation
from chest_cancer_detection import logger


STAGE_NAME = "Model Trainer Stage"

class EvaluationPipeline:

    def __init__(self):
        pass

    def main(self):
      config = ConfigurationManager()
      eval_config = config.get_evaluation_config()
      evaluation = Evaluation(eval_config)
      evaluation.evaluation()
      evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info("*******************************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e