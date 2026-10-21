import os as _os

_env = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), ".env")
if _os.path.exists(_env):  # minimal .env loader (no extra dependency); real env vars win
    for _line in open(_env):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _v.strip():
                _os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
