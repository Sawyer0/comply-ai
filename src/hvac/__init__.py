# Minimal stub for hvac so tests can patch hvac.Client without the real package.
class Client:  # pragma: no cover - used only for monkeypatching
    def __init__(self, *args, **kwargs):
        pass
