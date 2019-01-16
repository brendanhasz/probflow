
# TODO: import as module and use absolute imports
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from probflow import Variable


def test_variable_build():
    """Tests building a variable"""
    lala = Variable()
    lala._build(3)
