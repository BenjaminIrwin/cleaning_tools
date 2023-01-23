def get_batch(list, batch_size):
    # Iterate over a list with a batch size
    for i in range(0, len(list), batch_size):
        yield list[i:i + batch_size]