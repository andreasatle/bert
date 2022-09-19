"""
This module avoids creating __pycache__ when importing modules, and runs the workflow.
"""

if __name__ == "__main__":
    import sys

    sys.dont_write_bytecode = True
    import workflow

    workflow.run(1)
