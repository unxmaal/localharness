import re
def parse_duration(s):
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s or "")
    if not m or not any(m.groups()):
        raise ValueError(f"bad duration: {s!r}")
    h, mi, sec = (int(g or 0) for g in m.groups())
    return h * 3600 + mi * 60 + sec
