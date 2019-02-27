# Sets global variable notebook when line magic is called with "%torchbearer notebook"global notebook
notebook = {'nb': False}


def set_notebook(is_notebook):
    notebook['nb'] = is_notebook


try:
    from IPython.core.magic import register_line_magic
    @register_line_magic
    def torchbearer(line):
        if line == 'notebook':
            set_notebook(True)
        elif line == 'normal':
            set_notebook(False)
    set_notebook(True)
    del torchbearer  # Avoid scope issues

except NameError:
    pass


def is_notebook():
    return notebook['nb']