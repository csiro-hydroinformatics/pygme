""" Model objects factory """

from pygme.models import gr2m, gr4j, gr6j, lagroute, sac15, \
                    turcmezentsev, wapaba

MODEL_NAMES = ["gr2m", "g4j", "gr6j", "lagroute", "sac15", \
                "turcmezentsev", "wapaba"]


def model_factory(name, *args, **kwargs):
    """ Generate model objects.

        Parameters
        ----------
        name : str
            Model name
        *args : list, **kwargs : list, dict
            Model constructor kwargs.

        Returns
        -------
        model : pygme.model.Model
            Model
    """
    txt = "/".join(MODEL_NAMES)
    errmsg = f"Expected model name in {txt}, got {name}."
    assert name in MODEL_NAMES, errmsg

    if name == "gr2m":
        return gr2m.GR2M(*args, **kwargs)
    elif name == "gr4j":
        return gr4j.GR4J(*args, **kwargs)
    elif name == "gr6j":
        return gr6j.GR6J(*args, **kwargs)
    elif name == "lagroute":
        return lagroute.LagRoute(*args, **kwargs)
    elif name == "sac15":
        return sac15.SAC15(*args, **kwargs)
    elif  name == "turcmezentsev":
        return turcmezentsev.TurcMezentsev(*args, **kwargs)
    elif name == "wapaba":
        return wapaba.WAPABA(*args, **kwargs)


def calibration_factory(name, *args, **kwargs):
    """ Generate calibration object.

        Parameters
        ----------
        name : str
            Model name
        *args : list, **kwargs : list, dict
            Calibration object constructor kwargs.

        Returns
        -------
        calib : pygme.calibration.Calibration
            Calibration object.
    """
    txt = "/".join(MODEL_NAMES)
    errmsg = f"Expected model name in {txt}, got {name}."
    assert name in MODEL_NAMES, errmsg

    if name == "gr2m":
        return gr2m.CalibrationGR2M(*args, **kwargs)
    elif name == "gr4j":
        return gr4j.CalibrationGR4J(*args, **kwargs)
    elif name == "gr6j":
        return gr6j.CalibrationGR6J(*args, **kwargs)
    elif name == "lagroute":
        return lagroute.CalibrationLagRoute(*args, **kwargs)
    elif name == "sac15":
        return sac15.CalibrationSAC15(*args, **kwargs)
    elif  name == "turcmezentsev":
        return turcmezentsev.CalibrationTurcMezentsev(*args, **kwargs)
    elif name == "wapaba":
        return wapaba.CalibrationWAPABA(*args, **kwargs)

