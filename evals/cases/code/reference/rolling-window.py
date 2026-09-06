def windows(items, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    return [list(items[i:i + n]) for i in range(len(items) - n + 1)]
