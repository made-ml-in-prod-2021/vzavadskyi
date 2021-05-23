from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="KNeighborsClassifier")
    random_state: str = field(default=13)