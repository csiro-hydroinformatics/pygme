import imp

def has_c_module(name, raise_error=True):
    try:
        name = f"c_pygme_{name}"
        fp, pathname, description = imp.find_module(name)
        return True
    except ImportError:
        if raise_error:
            raise ImportError(f"C module c_pygme_{name} is "+\
                "not available, please run python setup.py build")
        else:
            return False


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
