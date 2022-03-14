from itertools import product as prod
import pytest

from pygme.factory import MODEL_NAMES, model_factory, calibration_factory, \
                            parameters_transform_factory
from pygme.calibration import ObjFunBCSSE


def test_model_factory():
    for model_name in MODEL_NAMES:
        model = model_factory(model_name)

def test_model_factory_ihacres():
    for shape in [0, 1, 2, 2.5]:
        model = model_factory(f"IHACRES{shape}")
        assert model.shapefactor == shape


def test_model_factory_gr2m():
    for r in [10, 60, 1000]:
        model = model_factory(f"GR2M{r}")
        assert model.Rcapacity == r


def test_calibration_factory():
    for model_name in MODEL_NAMES:
        model = calibration_factory(model_name)


def test_calibration_factory_ihacres():
    for shape in [0, 1, 2, 2.5]:
        calib = calibration_factory(f"IHACRES{shape}")
        assert calib.model.shapefactor == shape


def test_calibration_factory_gr2m():
    for r in [10, 60, 1000]:
        calib = calibration_factory(f"GR2M{r}")
        assert calib.model.Rcapacity == r


def test_calibration_factory_argument():
    objfun = ObjFunBCSSE(0.2)
    for model_name in MODEL_NAMES:
        calib = calibration_factory(model_name, objfun=objfun)
        assert hasattr(calib.objfun, "trans")
        assert calib.objfun.trans.lam == 0.2

def test_paramtrans_factory():
    for model_name in MODEL_NAMES:
        if model_name != "TurcMezentsev":
            f1, f2 = parameters_transform_factory(model_name)
        else:
            msg = "No transform available"
            with pytest.raises(ValueError, match=msg):
                f1, f2 = parameters_transform_factory(model_name)

