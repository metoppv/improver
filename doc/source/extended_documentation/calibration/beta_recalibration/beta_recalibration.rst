##################################
Beta recalibration
##################################


Beta recalibration is intended to be used for recalibrating the blended probabilistic output of several models, each of which has already been individually 
calibrated (for example, by reliability calibration or by the Rainforests calibration method). It is shown in `Ranjan & Gneiting, 2008`_
that, when blending probabilistic forecasts, even if each input is perfectly calibrated, the output is in general not perfectly calibrated. 
The authors also show that applying a recalibration to the blended output improves its reliabiilty, sharpness, and score on proper scoring metrics. 
IMPROVER implements the recalibration method studied in the 
article, namely a transformation given by the cumulative distribution function of the the beta distribution. 
The implementation here allows the `alpha` and `beta` parameters of the beta distribution to vary by forecast period.

.. _Ranjan & Gneiting, 2008: https://stat.uw.edu/sites/default/files/files/reports/2008/tr543.pdf

