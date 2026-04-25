import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import inspect
from core.methods.csrel import CSReLSelector

sig = inspect.signature(CSReLSelector.__init__)
print('CSReL __init__ 参数列表:')
for name, param in sig.parameters.items():
    if name == 'self':
        continue
    default = param.default if param.default != inspect.Parameter.empty else "(required)"
    print(f'  - {name}: {default}')
