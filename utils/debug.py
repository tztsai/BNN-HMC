import functools, re, inspect


def trace(fn):
    """A decorator that prints a function's name, its arguments, and its return
    values each time the function is called. For example,

    @trace
    def compute_something(x, y):
        # function body
    """
    indent = ''
    
    def log(message):
        print(indent + re.sub('\n', '\n' + indent, str(message)))

    @functools.wraps(fn)
    def wrapped(*args, **kwds):
        nonlocal indent
        reprs = [repr(e) for e in args]
        reprs += [repr(k) + '=' + repr(v) for k, v in kwds.items()]
        log('{0}({1})'.format(fn.__name__, ', '.join(reprs)) + ':')
        indent += '    '
        try:
            result = fn(*args, **kwds)
            indent = indent[:-4]
        except Exception as e:
            log(fn.__name__ + ' exited via exception')
            indent = indent[:-4]
            raise
        # Here, print out the return value.
        log('{0}({1}) -> {2}'.format(fn.__name__, ', '.join(reprs), result))
        return result
    return wrapped


def trace_all():
    frame = inspect.currentframe().f_back.f_locals
    for s, f in frame.items():
        if callable(f):
            globals()[s] = trace(f)