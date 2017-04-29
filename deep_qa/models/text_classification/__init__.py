from .classification_model import ClassificationModel
from .MultiClassificationModel import MultiClassificationModel

concrete_models = {  # pylint: disable=invalid-name
        'ClassificationModel': ClassificationModel,
        'MultiClassificationModel': MultiClassificationModel,
        }
