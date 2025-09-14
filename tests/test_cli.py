import os


def test_entrypoint():
    exit_status = os.system("re-forecast --help")
    assert exit_status == 0

