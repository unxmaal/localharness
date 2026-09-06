def human_bytes(n):
    if n < 1024:
        return f"{n} B"
    value = float(n)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
