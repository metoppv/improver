##################################
Beta recalibration
##################################

Beta recalibration is intended to be used for recalibrating the blended probabilistic output of several models, each of which has already been individually 
calibrated (for example, by reliability calibration or by the Rainforests calibration method). It is shown in `Ranjan & Gneiting, 2008`_ 
that, when blending probabilistic forecasts, even if each input is perfectly calibrated, the output is in general not perfectly calibrated. 
The authors also show that applying a recalibration to the blended output improves its reliability, sharpness, and score on proper scoring metrics. 
IMPROVER implements the recalibration method studied in the 
article, namely a transformation given by the cumulative distribution function of the beta distribution. This is a natural choice of transformation 
function for probabilities because it is a monotone-increasing function that maps the interval `[0, 1]` onto itself.
The implementation here allows the `alpha` and `beta` parameters of the beta distribution to vary by forecast period.

It is recommended that the `alpha` and `beta` parameters be chosen to optimise the desired metric (for example, the continuous rank
probability score) over some training period. Specifically, this could be done as follows:

1. Obtain blended forecast output, along with ground truth data from observations or analyses, for the training period.
2. Implement the loss function. The loss function should have parameters `alpha` and `beta` and return the loss over the training period when the probabilistic forecast is transformed by the CDF of the beta distribution function. A suggested loss function is the CRPS calculated from the thresholded probability forecast.  In this case, the loss function should transform the input blended probabilities by the CDF of the beta distribution (which is available in `scipy.stats`), then calculate the CRPS of the transformed probability forecast against the ground truth.
3. Use `scipy.optimize.minimize` to find the parameters `alpha` and `beta` that minimise the loss.

Alternatively, one could jointly optimise the blending weights and the parameters of the beta calibration. This may yield better results, but 
is more complex.

.. _Ranjan & Gneiting, 2008: https://stat.uw.edu/sites/default/files/files/reports/2008/tr543.pdf

