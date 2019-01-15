
# TODO: import as module and use absolute imports
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from probflow import Variable

lala = Variable()

lala._build(3)