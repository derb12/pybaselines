def _setup_window(data, half_window, **pad_kwargs):
    return pad_edges(np.asarray(data), half_window, **pad_kwargs)
