**Constructing percentile and probabilities from the location and scale parameters**

In order to construct percentiles or probabilities from the location or scale
parameter, the distribution for the resulting output needs to be selected. For
use with the outputs from EMOS, where it has been assumed that the outputs
from minimising the CRPS follow a particular distribution, then the same
distribution should be selected, as used for the CRPS minimisation. The
conversion to percentiles and probabilities from the location and scale
parameter relies upon functionality within scipy.stats.
