""" Model objects factory """

from pygme.models import gr2m, gr4j, gr6j, lagroute, sac15, \
                    turcmezentsev, wapaba, ihacres

MODEL_NAMES = ["GR2M", "GR4J", "GR6J", "LagRoute", "SAC15", \
                "TurcMezentsev", "WAPABA", "IHACRES"]


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

    if name == "GR2M":
        return gr2m.GR2M(*args, **kwargs)
    elif name == "GR4J":
        return gr4j.GR4J(*args, **kwargs)
    elif name == "GR6J":
        return gr6j.GR6J(*args, **kwargs)
    elif name == "LagRoute":
        return lagroute.LagRoute(*args, **kwargs)
    elif name == "SAC15":
        return sac15.SAC15(*args, **kwargs)
    elif  name == "TurcMezentsev":
        return turcmezentsev.TurcMezentsev(*args, **kwargs)
    elif name == "WAPABA":
        return wapaba.WAPABA(*args, **kwargs)
    elif name == "IHACRES":
        return ihacres.IHACRES(*args, **kwargs)


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

    if name == "GR2M":
        return gr2m.CalibrationGR2M(*args, **kwargs)
    elif name == "GR4J":
        return gr4j.CalibrationGR4J(*args, **kwargs)
    elif name == "GR6J":
        return gr6j.CalibrationGR6J(*args, **kwargs)
    elif name == "LagRoute":
        return lagroute.CalibrationLagRoute(*args, **kwargs)
    elif name == "SAC15":
        return sac15.CalibrationSAC15(*args, **kwargs)
    elif  name == "TurcMezentsev":
        return turcmezentsev.CalibrationTurcMezentsev(*args, **kwargs)
    elif name == "WAPABA":
        return wapaba.CalibrationWAPABA(*args, **kwargs)
    elif name == "IHACRES":
        return ihacres.CalibrationIHACRES(*args, **kwargs)

