from typing import Any
import numpy as np


def accuracy(pred: Any, true: Any) -> Any:
    return np.mean(pred == true)
