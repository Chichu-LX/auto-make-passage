"""Shared runtime package for the isolated research agents.

The package deliberately avoids eager imports so individual Agent packages can
depend on schemas/providers without creating an import cycle through Runtime.
"""

__all__: list[str] = []
