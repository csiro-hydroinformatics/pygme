from itertools import product as prod
import pytest

from pygme.factory import MODEL_NAMES, model_factory, calibration_factory
from pygme.calibration import ObjFunBCSSE


def test_model_factory():
    for model_name in MODEL_NAMES:
        model = model_factory(model_name)

def test_calibration_factory():
    for model_name in MODEL_NAMES:
        model = calibration_factory(model_name)

def test_calibration_factory_argument():
    objfun = ObjFunBCSSE(0.2)
    for model_name in MODEL_NAMES:
        model = calibration_factory(model_name, objfun=objfun)

