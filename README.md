# spsm-GLKF

Copyright (©) 2019, Mattia F. Pagnotta and David Pascucci.

The codes provide a MATLAB implementation of spsm-GLKF, a regularized and smoothed adaptive algorithm based on the Kalman filter, which allows estimating time-varying multivariate autoregressive (tv-MVAR) models.
The algorithm is an extension of the General Linear Kalman Filter (GLKF) proposed in (Milde et al., Neuroimage, 2010). This new extension of GLKF incorporates ℓ1 norm penalties for tv-MVAR coefficients selection, and a smoothing procedure using the Rauch–Tung–Striebel (RTS) fixed-interval smoother (Rauch et al., AIAA journal, 1965).

For more detailed description of the method, please refer to:

Pagnotta, M.F., Plomp, G., & Pascucci, D. (2019). A regularized and smoothed General Linear Kalman Filter for more accurate estimation of time-varying directed connectivity*. 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 611-615. https://doi.org/10.1109/EMBC.2019.8857915


Useful references:

Milde, T., Leistritz, L., Astolfi, L., Miltner, W. H., Weiss, T., Babiloni, F., & Witte, H. (2010). A new Kalman filter approach for the estimation of high-dimensional time-variant multivariate AR models and its application in analysis of laser-evoked brain potentials. Neuroimage, 50(3), 960-969. https://doi.org/10.1016/j.neuroimage.2009.12.110

Rauch, H. E., Striebel, C. T., & Tung, F. (1965). Maximum likelihood estimates of linear dynamic systems. AIAA journal, 3(8), 1445-1450. https://doi.org/10.2514/3.3166



This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
