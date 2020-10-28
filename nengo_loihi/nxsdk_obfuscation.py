"""
NengoLoihi is a publicly viewable project. NxSDK is not. In order to be able
to construct classes and call function that are part of NxSDK without making
NengoLoihi private, we need a mechanism to obfuscate those parts of
NengoLoihi.

It is not possible to encrypt / compile these parts of NengoLoihi such that
they are unrecoverable; instead, the goal here is to prevent shoulder surfing,
meaning that anyone wishing to see certain strings that are part of the
NxSDK API will at least have to take some steps to obtain those strings,
rather than being able to see them directly.
"""

import base64
import importlib


def obfuscate(obfs_str):
    return base64.b64encode(obfs_str.encode())


def deobfuscate(obfs_str, cast=None, debug=None):
    d_str = base64.b64decode(obfs_str).decode()

    if cast is None:
        result = d_str
    elif isinstance(cast, str) and cast.startswith("list"):
        d_str = d_str.strip("[]").split(", ")
        if cast.endswith("int"):
            result = [int(x) for x in d_str]
        else:
            result = [float(x) for x in d_str]
    else:
        result = cast(d_str)

    if debug is not None:
        assert result == debug

    return result


d = deobfuscate


def d_import(pkg, attr=None):
    result = importlib.import_module(deobfuscate(pkg))
    if attr is not None:
        result = getattr(result, d(attr))

    return result


def d_get(obj, *attrs):
    result = obj
    for attr in attrs:
        result = getattr(result, deobfuscate(attr))

    return result


def d_set(obj, *attrs, val):
    result = obj
    for attr in attrs[:-1]:
        result = getattr(result, deobfuscate(attr))

    attr = attrs[-1]
    assert hasattr(result, deobfuscate(attr))
    setattr(result, deobfuscate(attr), val)


def d_func(obj, *attrs, kwargs=None):
    if kwargs is None:
        kwargs = {}
    else:
        kwargs = {deobfuscate(k): v for k, v in kwargs.items()}
    func = d_get(obj, *attrs)
    return func(**kwargs)
