import sys
if sys.version_info[0] < 3:
    def set_doc(inner, doc):
        return None  # Not simple to do in Python 2.7 so we can leave it for now, just build docs with Python 3+
else:
    def set_doc(inner, doc):
        inner.__doc__ = doc


def cite(bibtex):
    """A decorator which adds a reference to the **Google style** docstring of the given object. The ``Args:`` or
    ``Returns:`` line is then prepended with the given bibtex string at runtime. Otherwise, the last line is used.

    Args:
        bibtex (str): The bibtex string to insert

    Returns:
        The decorator
    """
    def decorator(inner):
        doc = inner.__doc__.split('\n')
        i = 0
        s = 0
        for line in doc:
            sline = line.strip()
            if sline == 'Args:' or sline == 'Returns:':
                for char in line:
                    if char == ' ':
                        s += 1
                break
            i += 1

        spaces = ' ' * (s + 4)
        to_insert = ' ' * s + '::\n\n' + spaces
        to_insert += bibtex.strip().replace('\n', '\n' + spaces).rstrip()

        doc.insert(i, '')
        doc.insert(i, to_insert)
        set_doc(inner, '\n'.join(doc))
        return inner
    return decorator
