import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # add software/ to path
# The above line must be before any imports from the software/ directory
# to ensure the correct modules are imported.
