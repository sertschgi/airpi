_USE_TPU = True

def get_USE_TPU():
    return _USE_TPU

def set_USE_TPU(useTpu):
    global _USE_TPU
    _USE_TPU = useTpu
