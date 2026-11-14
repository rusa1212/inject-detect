"""패키지 import 스모크 테스트."""

import promptshield


def test_import_package() -> None:
    assert isinstance(promptshield.__version__, str)
    assert promptshield.__version__
