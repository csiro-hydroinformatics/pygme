""" Pygme objects factory """
import re
from pygme.models import gr2m, gr4j, gr6j, lagroute, sac15, \
                    turcmezentsev, wapaba, ihacres
from pygme import calibration

MODEL_NAMES = ["GR2M", "GR4J", "GR6J", "LagRoute", "SAC15", \
                "TurcMezentsev", "WAPABA", "IHACRES"]

def check_model(name):
    txt = "/".join(MODEL_NAMES)
    errmsg = f"Expected model name in {txt}, got {name}."
    assert name in MODEL_NAMES or name.startswith("IHACRES") \
                    or name.startswith("GR2M"), errmsg


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
    check_model(name)

    if name.startswith("GR2M"):
        if name != "GR2M":
            kwargs["Rcapacity"] = gr2m.get_rcapacity(name)
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
    elif name.startswith("IHACRES"):
        if name != "IHACRES":
            kwargs["shapefactor"] = ihacres.get_shapefactor(name)
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
    check_model(name)

    if name.startswith("GR2M"):
        if name != "GR2M":
            kwargs["Rcapacity"] = gr2m.get_rcapacity(name)
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
    elif name.startswith("IHACRES"):
        if name != "IHACRES":
            kwargs["shapefactor"] = ihacres.get_shapefactor(name)
        return ihacres.CalibrationIHACRES(*args, **kwargs)


def parameters_transform_factory(name):
    """ Generate parameter transformation objects.

        Parameters
        ----------
        name : str
            Model name

        Returns
        -------
        true2trans : function
            Function to transform true parameters to transformed.
        trans2true : function
            Function to transform transformed parameters to true.
    """
    check_model(name)

    if name.startswith("GR2M"):
        return gr2m.gr2m_true2trans, gr2m.gr2m_trans2true
    elif name == "GR4J":
        return gr4j.gr4j_true2trans, gr4j.gr4j_trans2true
    elif name == "GR6J":
        return gr6j.gr6j_true2trans, gr6j.gr6j_trans2true
    elif name == "LagRoute":
        return lagroute.lagroute_true2trans, lagroute.lagroute_trans2true
    elif name == "SAC15":
        return sac15.sac15_true2trans, sac15.sac15_trans2true
    elif  name == "TurcMezentsev":
        raise ValueError("No transform available")
    elif name == "WAPABA":
        return wapaba.wapaba_true2trans, wapaba.wapaba_trans2true
    elif name.startswith("IHACRES"):
        return ihacres.ihacres_true2trans, ihacres.ihacres_trans2true


def objfun_factory(name, *args, **kwargs):
    """ Generate objective function objects.

        Parameters
        ----------
        name : str
            Objective function name as follows:
            * bcx.y: Box-Cox transform SSE with exponent x.y (e.g. bc0.5)
            * biasbcx.y: Box-Cox transform SSE with bias constraint.
            * kge: KGE objective function.
            * sse: Sum of squared error.

        *args : list, **kwargs : list, dict
            Model constructor kwargs.

        Returns
        -------
        objfun : pygme.calibration.ObjFun
            Objective function
    """
    if name.startswith("biasbc"):
        lam = float(re.sub("^biasbc", "", name))
        objfun = calibration.ObjFunBiasBCSSE(lam, *args, **kwargs)

    elif name.startswith("bc"):
        lam = float(re.sub("^bc", "", name))
        objfun = calibration.ObjFunBCSSE(lam, *args, **kwargs)

    elif name == "kge":
        objfun = calibration.ObjFunKGE()

    elif name == "sse":
        objfun = calibration.ObjFunSSE()

    else:
        raise ValueError(f"Objective function {name} not recognised")

    return objfun
