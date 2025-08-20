import importlib

__version__ = "1.4"

def has_c_module(name, raise_error=True):
    name = f"c_pygme_{name}"
    out = importlib.util.find_spec(name)

    if not out is None:
        return True
    else:
        if raise_error:
            raise ImportError(f"C module c_pygme_{name} is "+\
                "not available, please run python setup.py build")
        else:
            return False

