from .bbrbm import BBRBM
from .gbrbm import GBRBM
from .bbrbm_temp import BBRBMTEMP
# default RBM
RBM = BBRBM

__all__ = [RBM, BBRBM, GBRBM,BBRBMTEMP]
