def chunks(data, size):
    if size <= 0:
        raise ValueError("size must be positive")
    return [data[i:i + size] for i in range(0, len(data), size)]
