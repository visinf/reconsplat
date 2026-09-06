import os

def _DEBUGGING(ENV_VAR: str = 'DEBUG', TRUE_VALUE: int | str = '1') -> bool:
    if ENV_VAR in os.environ and os.environ[ENV_VAR] == TRUE_VALUE:
        return True
    return False

     
    