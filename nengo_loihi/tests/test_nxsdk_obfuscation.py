import pytest

from nengo_loihi.nxsdk_obfuscation import (
    d_func,
    d_get,
    d_import,
    d_set,
    deobfuscate,
    obfuscate,
)


def test_obfuscate_deobfuscate():
    my_str = "Test test"
    assert obfuscate(my_str) != my_str
    assert deobfuscate(obfuscate(my_str)) == my_str

    my_list = [0, 1, 2]
    assert obfuscate(str(my_list)) != my_list
    assert deobfuscate(obfuscate(str(my_list)), "list_int") == my_list

    my_list = [0.5, 1.5, 2.5]
    assert obfuscate(str(my_list)) != my_list
    assert deobfuscate(obfuscate(str(my_list)), "list_float") == my_list

    my_int = 5
    assert obfuscate(str(my_int)) != my_int
    assert deobfuscate(obfuscate(str(my_int)), int) == my_int

    with pytest.raises(AssertionError):
        deobfuscate(obfuscate("0"), debug="1")
    deobfuscate(obfuscate("0"), debug="0")


def test_d_import():
    imported0 = d_import(
        obfuscate("nengo_loihi.emulator.interface"), attr=obfuscate("EmulatorInterface")
    )
    # pylint: disable=import-outside-toplevel
    from nengo_loihi.emulator.interface import EmulatorInterface as imported1

    assert imported0 is imported1


def test_d_get_set():
    class TestClass:
        pass

    obj = TestClass()

    obj.attr0 = TestClass()
    obj.attr0.attr1 = "test"

    assert d_get(obj, obfuscate("attr0"), obfuscate("attr1")) == "test"

    # error if trying to set a new attribute
    with pytest.raises(AssertionError):
        d_set(obj, obfuscate("attr0"), obfuscate("attr2"), val="test2")

    obj.attr0.attr2 = None
    d_set(obj, obfuscate("attr0"), obfuscate("attr2"), val="test2")
    assert obj.attr0.attr2 == "test2"


def test_d_func():
    class TestClass:
        def test_func(self, val=None):
            return val

    obj = TestClass()
    assert (
        d_func(obj, obfuscate("test_func"), kwargs={obfuscate("val"): "test"}) == "test"
    )
