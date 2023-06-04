class LazyCall:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        if not hasattr(self, "_lazy_return_value"):
            self._lazy_return_value = self.function(*self.args, **self.kwargs)
        return self._lazy_return_value
