# Sets global variable notebook when line magic is called with "%torchbearer notebook"global notebook
notebook = {'nb': False}


def set_notebook(is_notebook):
    notebook['nb'] = is_notebook


def torchbearer(line):
    if line == 'notebook':
        set_notebook(True)
    elif line == 'normal':
        set_notebook(False)


try:
    import IPython.core.magic
    torchbearer = IPython.core.magic.register_line_magic(torchbearer)
    set_notebook(True)
except (NameError, ImportError) as e:
    pass


def is_notebook():
    return notebook['nb']
