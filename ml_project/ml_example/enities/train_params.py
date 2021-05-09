from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="KNeighborsClassifier")
    random_state: str = field(default=13)
    leaf_size: str = field(default=1)
    n_neighbors: str = field(default=14)
    p: str = field(default=2)
    weights: str = field(default='uniform')