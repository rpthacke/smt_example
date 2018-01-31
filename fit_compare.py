"""syncing up interpolators."""
import numpy as np
import matplotlib.pyplot as plt
import suckit

import brokeit
from smt.methods import RMTC, RMTB
from lib.thrust_data import thrust_problem, make_plot
from lib.interpolate import RegularGridInterpolator
n = 80
mn, alt, thrust, pts_test = thrust_problem(n, norm=False)

# flatten training data

a, b = np.meshgrid(mn, alt, indexing='ij')
xt = np.array([a.ravel(), b.ravel()]).T
yt = thrust.flatten()

# -------------------- smt ----------------------------------
xlimits = np.array([[mn[0], mn[-1]], [alt[0], alt[-1]]])

sm = RMTC(xlimits=xlimits, print_global=False, num_elements=10,
          reg_dv=1e-15, reg_cons=1e-15, approx_order=1)
sm.set_training_values(xt, yt)
sm.train()
img1 = sm.predict_values(pts_test).reshape(n, n)

validate = sm.predict_values(xt)
err = np.linalg.norm(yt - validate) / np.linalg.norm(yt)

g1 = sm.predict_derivatives(pts_test, 0)
g2 = sm.predict_derivatives(pts_test, 1)
gradient1 = np.array([g1, g2]).T.reshape(n, n, 2)
make_plot(img1, gradient1, mn, alt, thrust, "RMTC, rel.err=%2.5f" % err)

# --------------------- sgi ----------------------------------
interp_cubic = RegularGridInterpolator([mn, alt], thrust, method='cubic')
img2 = interp_cubic(pts_test, compute_gradients=False).reshape(n, n)

validate = interp_cubic(xt, compute_gradients=False)
err = np.linalg.norm(yt - validate) / np.linalg.norm(yt)

gradient2 = interp_cubic.gradient(pts_test).reshape(n, n, 2)
make_plot(img2, gradient2, mn, alt, thrust, "RGI, rel.err=%2.5f" % err)

plt.show()
