# ============================
# TESTBED
# ============================

using PyCall
pushfirst!(PyVector(pyimport("sys")."path"),"/home/sbae/automotive-control-temporary/python/nnmpc/sgan")
sganPredictor = pyimport("sgan.predictor")
model_path = "/home/sbae/automotive-control-temporary/python/nnmpc/sgan/models/sgan-models/eth_8_model.pt"
predictor = sganPredictor.Predictor(model_path)

data = [  [ 1,  1.000e+00,  8.460e+00,  3.590e+00],
          [ 1,  2.000e+00,  1.364e+01,  5.800e+00],
          [ 2,  1.000e+00,  9.570e+00,  3.790e+00],
          [ 2,  2.000e+00,  1.364e+01,  5.800e+00],
          [ 3,  1.000e+00,  1.067e+01,  3.990e+00],
          [ 3,  2.000e+00,  1.364e+01,  5.800e+00],
          [ 4,  1.000e+00,  1.173e+01,  4.320e+00],
          [ 4,  2.000e+00,  1.209e+01,  5.750e+00],
          [ 5,  1.000e+00,  1.281e+01,  4.610e+00],
          [ 5,  2.000e+00,  1.137e+01,  5.800e+00],
          [ 6,  1.000e+00,  1.281e+01,  4.610e+00],
          [ 6,  2.000e+00,  1.031e+01,  5.970e+00],
          [ 7,  1.000e+00,  1.194e+01,  6.770e+00],
          [ 7,  2.000e+00,  9.570e+00,  6.240e+00],
          [ 8,  1.000e+00,  1.103e+01,  6.840e+00],
          [ 8,  2.000e+00,  8.730e+00,  6.340e+00]]

@elapsed predictor.predict(data)
pred_traj = predictor.predict(data)
