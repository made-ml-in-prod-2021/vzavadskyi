import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig, OmegaConf
from .split_params import SplittingParams
from .train_params import TrainingParams
from .feature_params import FeatureParams

@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    report_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)



def read_training_pipeline_params(cfg: DictConfig) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    return schema.load(yaml.safe_load(OmegaConf.to_yaml(cfg)))
