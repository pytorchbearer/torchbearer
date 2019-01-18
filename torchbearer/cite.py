def cite(bibtex):
    """A decorator which adds a reference to the docstring of the given object. The docstring must contain ':bib:' which
    is then replaced with the given bibtex string at runtime.

    Args:
        bibtex (str): The bibtex string to insert

    Returns:
        The decorator
    """
    to_insert = '::\n\n' + ' '*8
    to_insert += bibtex.strip().replace('\n', '\n' + ' '*8).rstrip()

    def decorator(inner):
        inner.__doc__ = inner.__doc__.replace(':bib:', to_insert)
        return inner
    return decorator
