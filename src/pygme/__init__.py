import importlib

__version__ = "1.4"


def has_c_module(name, raise_error=True):
    m_name = f"c_pygme_{name}"
    out = importlib.util.find_spec(m_name)

    if out is not None:
        return True
    else:
        if raise_error:
            errmsg = f"C module {m_name} is not available."\
                     + " Please run 'pip install -e .'"
            raise ImportError(errmsg)
        else:
            return False
