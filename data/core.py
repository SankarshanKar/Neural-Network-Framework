import numpy as np
import inspect

def init(dot_precision_workaround=True, default_dtype='float32', random_seed=0):
    methods_to_enclose = [
        [np, 'zeros', False, 1],
        [np.random, 'randn', True, None],
        [np, 'eye', False, 1],
    ]

    if dot_precision_workaround:
        orig_dot = np.dot
        def dot(*args, **kwargs):
            return orig_dot(*[a.astype('float64') for a in args], **kwargs).astype('float32')
        np.dot = dot
    else:
        methods_to_enclose.append([np, 'dot', True, None])

    if default_dtype:
        for method in methods_to_enclose:
            enclose(method, default_dtype)

    if random_seed is not None:
        np.random.seed(random_seed)

def enclose(method, default_dtype):
    method.append(getattr(*method[:2]))
    def enclosed_method(*args, **kwargs):
        if method[2]:
            return method[4](*args, **kwargs).astype(default_dtype)
        else:
            if len(args) <= method[3] and 'dtype' not in kwargs:
                kwargs['dtype'] = default_dtype
            return method[4](*args, **kwargs)

    setattr(*method[:2], enclosed_method)
