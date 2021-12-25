import numpy as np
import scipy
import math
import time

"""
Part I
"""

" Covariance functions. For each function we implemented a 'rescaled' function "
" which has the advantage of saving one division by param(k) , 1\leq k \leq d. "
" This has been done to optimize the computation times. "
" Here we use vectorization to replace 'for' iteration "

"we need an efficiency method to compute the inverse of matrix, here we choose cholesky"


def mat_inverse(A):
    L = np.matrix(scipy.linalg.cholesky(A, lower=False))
    'Use lapack dtrtri to compute the inverse of upper triangular matirx'
    L_inv = scipy.linalg.lapack.dtrtri(L)
    L_inv = np.matrix(L_inv[0])
    return L_inv * L_inv.T
    # return np.linalg.inv(A)


" Scale factors "


def CovScalingFactor(Type):
    if (Type == 'gauss'):
        return math.sqrt(2) / 2
    elif (Type == 'matern3_2'):
        return math.sqrt(3)
    elif (Type == 'matern5_2'):
        return math.sqrt(5)
    else:
        return 1


def CovGauss_scale(x1, x2, var):
    " x1, x2 are row vectors "
    " var is the variance, which is sigma ** 2 "
    dist = np.dot((x1 - x2), (x1 - x2).T)
    "return math.exp(-dist) * var"
    return math.exp(-dist) * var


def CovExp_scale(x1, x2, var):
    dist = np.sum(np.abs(x1 - x2))
    return math.exp(-dist) * var


def CovMatern3_2_scale(x1, x2, var):
    dist = np.sum(np.abs(x1 - x2))
    return math.exp(-dist) * var * (1 + dist)


def CovMatern5_2_scale(x1, x2, var):
    dist = np.sum(np.abs(x1 - x2))
    return math.exp(-dist) * var * (1 + dist + (pow(dist, 2)) / 3)


" use func_dict to choose different kernels "
func_dict = {"gauss": CovGauss_scale, "exp": CovExp_scale,
             "matern3_2": CovMatern3_2_scale, "matern5_2": CovMatern5_2_scale}


def CovFunction(Type):
    return func_dict.get(Type, 'There is no such kernel')


" Compute kernal matrix on the same matrix "


def CovMatrix_scale(x, var, Type):
    " result matrix is n * n "
    result = np.zeros((x.shape[0], x.shape[0]), dtype='float64')

    " determine kernel function "
    cov_function = CovFunction(Type)

    " x is a n * d matrix "
    for i in range(x.shape[0]):
        for j in range(i):
            result[i, j] = result[j, i] = cov_function(x[i, :], x[j, :], var)
        result[i, i] = var

    return np.matrix(result)


" Compute kernal matrix between different matries "


def CovMat1Mat2_scale(x1, x2, var, Type):
    " result matrix is n1 * n2 "
    result = np.zeros((x1.shape[0], x2.shape[0]), dtype='float64')

    " determine kernel function "
    cov_function = CovFunction(Type)

    " x1 is a n1 * d matrix, x2 is a n2 * d matrix "
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            result[i, j] = cov_function(x1[i, :], x2[j, :], var)

    return np.matrix(result)


" test "
"""x = np.array([[1, 2], [2, 3]])
result = CovMatrix_scale(x, 3, 'gauss')"""

"""
Part II
"""

"fast_pred function for computing predictions based on our approximate best linear predictor"
"we implement it with heteroscedastic noise"
"assume that our noise satisfies centered normal distribution with r(x) variance"


def fast_pred_noise(X, Y, n, d, gp, gpsize, N, x, q, var_noise,
              y_noise, covtype, var, param, nuggetfactor):
    """
    // X: n*d design matrix
    // Y: n-dimensional response vector
    // gp: n-dimensional vector containing the group number of all n points
    // gpsize: N-dimensional vector with the sizes of each group
    // N: total number of groups
    // x: q*d matrix of points where the prediction is computed:
    // q: number of points where a prediction is computed
    // var_noise: n dimensional vector containing the variance of noise of design points
    // y_noise: n dimensional vector containing value of noise sampled by normal distribution
    // covtype: the covariance name. Current possible choices are: "gauss", "exp", "matern3_2", "matern5_2"
    // l: d-dimensional vector of length scale
    // var: variance parameter
    // nuggetfactor: A factor which inflates variances when numerical issues arise. If the value of 1 does not work, the user should try 1.00000001
    """

    "(i) Declaration of the important variables"

    "In python, we can use list to replace vector of matrix"
    points_group = [0] * N
    points_response_noise = [0] * N
    Ki_inv_ki = [0] * N
    K_Mx = [0] * q
    "noise matrix list"
    D_x = [0] * N

    "In python, np.array gives array variables, we need to convert them into matrix"
    "the origin type of element in np.matrix is int32, we need to convert it to float"

    "covariance between group i and group j when a prediction is computed"

    k_Mx = np.matrix(np.zeros((N, q)), dtype='float64')
    "Y_hat is the transpose of M(x)"
    Y_hat = np.matrix(np.zeros((N, q)), dtype='float64')
    Xscale = np.zeros((n, d), dtype='float64')
    xscale = np.zeros((q, d), dtype='float64')
    weights = np.matrix(np.zeros((N, q)), dtype='float64')

    "ki_x, tki_x, Kij, Ki, Ki_inv is not sure, we'll define them later"

    predmean = np.zeros(q, dtype='float64')
    predsd2 = np.zeros(q, dtype='float64')
    "the number of points in each group"
    currentrow = np.zeros(N, dtype='int32')
    scalingfactor = CovScalingFactor(covtype)

    "(ii) Initialization of some matries"

    "rescale parameters"
    paramscale = param / scalingfactor

    "rescale x and X"
    for j in range(d):
        Xscale[:, j] = X[:, j] / paramscale[j]
        xscale[:, j] = x[:, j] / paramscale[j]

    "Initialization of the N matrices containing the points of group 1, 2, ..., N"
    for i in range(N):
        points_group[i] = np.zeros((gpsize[i], d), dtype='float64')
        points_response_noise[i] = np.zeros(gpsize[i], dtype='float64')
        D_x[i] = np.matrix(np.zeros((gpsize[i], gpsize[i]), dtype='float64'))

    "Initialization of K_Mx, k_xx"
    for i in range(q):
        K_Mx[i] = np.zeros((N, N), dtype='float64')
        K_Mx[i] = np.matrix(K_Mx[i])

    "Construction of the N matrices containing the points of group 1, 2, ..., N"
    for i in range(n):
        gp_i = gp[i]
        points_group[gp_i][currentrow[gp_i], :] = Xscale[i, :]
        "Now the points_response should add noise term"
        points_response_noise[gp_i][currentrow[gp_i]] = Y[i] + y_noise[i]
        "Construction of D_x containing the variance matrix of each group"
        D_x[gp_i][currentrow[gp_i], currentrow[gp_i]] = var_noise[i]
        "when point is in group gp_i, add 1"
        currentrow[gp_i] = currentrow[gp_i] + 1

    "(iii) Main loop of the algorithm"
    t1 = time.time()
    for i in range(N):
        Ki = CovMatrix_scale(points_group[i], var, covtype)
        "the Ki_inv will be replaced by inverse with noise, which is (k(Xi,Xi) + Di)^{-1}"
        "we need add a nugget matrix to avoid the singular condition"
        # Ki_D_inv = mat_inverse(Ki + D_x[i])
        nugget = 0.000001 * np.matrix(np.identity(gpsize[i]))
        Ki_D_inv = mat_inverse(Ki + D_x[i] + nugget)

        "ki_x is a ni*q matrix, that is, k(Xi, x)"
        ki_x = CovMat1Mat2_scale(points_group[i], xscale, var, covtype)
        tki_x = ki_x.T

        "We just use simple Kriging method"

        "k(x,Xi)(k(Xi,Xi) + Di)^{-1}"
        Ki_inv_ki[i] = Ki_D_inv * ki_x
        "M_{yita, i}(x) = k(x,Xi)(k(Xi,Xi) + Di)^{-1}(Y(Xi) + ksi(Xi))"
        Y_hat[i, :] = points_response_noise[i] * Ki_inv_ki[i]

        "j denotes the j-th point to predict"
        for j in range(q):
            k_Mx[i, j] = np.dot(tki_x[j, :], Ki_inv_ki[i][:, j])
            K_Mx[j][i, i] = nuggetfactor * k_Mx[i, j]
    t2 = time.time()
    t_test = t2 - t1

    "(iv) Second loop of the off-diagonal elements"
    "off-diagonal elements of k_Mx"
    t1 = time.time()
    for i in range(N):
        for j in range(i):
            "Kij is k(Xi, Xj)"
            Kij = CovMat1Mat2_scale(points_group[i], points_group[j], var, covtype)

            for k in range(q):
                K_Mx[k][i, j] = Ki_inv_ki[i][:, k].T * Kij * Ki_inv_ki[j][:, k]
                K_Mx[k][j, i] = K_Mx[k][i, j]
    t2 = time.time()
    t_test_1 = t2 - t1

    "(v) Conclusion of the algorithm"
    for i in range(q):
        nugget1 = 0.000001 * np.matrix(np.identity(N))
        K_Mx_inv = mat_inverse(K_Mx[i] + nugget1)
        # K_Mx_inv = mat_inverse(K_Mx[i])
        k_Mxi = k_Mx[:, i]

        "weights is coefficients of M_A, specifically, alpha.T"
        weights[:, i] = K_Mx_inv * k_Mxi

        predmean[i] = k_Mxi.T * K_Mx_inv * Y_hat[:, i]
        predsd2[i] = max(0, var - k_Mxi.T * K_Mx_inv * k_Mxi)

    "(vi) Returns the final result"
    Result = {}
    "M_A(x)"
    Result['m_A'] = predmean
    Result['v_A'] = predsd2
    Result['K_Mx'] = K_Mx
    Result['k_Mx'] = k_Mx
    "M(x)"
    Result['m'] = Y_hat
    Result['alpha'] = weights

    return Result


"""
Part III fast_LOOerror function for computing cross validation errors
"""

def fast_LOOerror(X, Y, n, d, gp, gpsize, N, indices, q, var_noise,
              y_noise, covtype, var, param, nuggetfactor):
    "indices: vector of size n with a 1 in position i if the LOO cross-validation of point i is computed"
    "here q is new parameter denoting the number of test points rather than predict points"
    "(1) Declaration of the important variables"

    points_group = [0] * N
    points_response_noise = [0] * N
    Ki_inv_ki = [0] * N
    K_Mx = [0] * q
    "noise matrix list"
    D_x = [0] * N

    k_Mx = np.matrix(np.zeros((N, q)), dtype='float64')
    "Y_hat is the transpose of M(x)"
    Y_hat = np.matrix(np.zeros((N, q)), dtype='float64')
    Xscale = np.zeros((n, d), dtype='float64')
    xscale = np.zeros((q, d), dtype='float64')
    weights = np.matrix(np.zeros((N, q)), dtype='float64')

    predmean = np.zeros(q, dtype='float64')
    predsd2 = np.zeros(q, dtype='float64')
    currentrow = np.zeros(N, dtype='int32')
    response_x = np.zeros(q, dtype='float64')
    LOOerror = np.zeros(q, dtype='float64')
    group_x = np.zeros(q, dtype='int32')
    position_in_group_x = np.zeros(q, dtype='int32')

    scaling_factor = CovScalingFactor(covtype)

    "(2) Initialization of some matrices"

    param_scale = param / scaling_factor

    "rescale X"
    for j in range(d):
        Xscale[:, j] = X[:, j] / param_scale[j]

    "Building xscale, the points where we predict are a subset of the points from X"
    current_index = 0
    for i in range(n):
        if indices[i] == 1:
            xscale[current_index, :] = Xscale[i, :]
            group_x[current_index] = gp[i]
            response_x[current_index] = Y[i] + y_noise[i]

            current_index = current_index + 1

    for i in range(N):
        "initialize D_x, points_group and point_response_noise"
        points_group[i] = np.zeros((gpsize[i], d), dtype='int32')
        points_response_noise[i] = np.zeros(gpsize[i], dtype='float64')
        D_x[i] = np.matrix(np.zeros((gpsize[i], gpsize[i]), dtype='float64'))

    for i in range(q):
        K_Mx[i] = np.matrix(np.zeros((N, N), dtype='float64'))

    current_index = 0
    for i in range(n):
        gp_i = gp[i]
        points_group[gp_i][currentrow[gp_i], :] = Xscale[i, :]
        points_response_noise[gp_i][currentrow[gp_i]] = Y[i] + y_noise[i]
        if indices[i] == 1:
            position_in_group_x[current_index] = currentrow[gp_i]
            current_index = current_index + 1
        D_x[gp_i][currentrow[gp_i], currentrow[gp_i]] = var_noise[i]
        currentrow[gp_i] = currentrow[gp_i] + 1

    "(3) First important loop"

    "The loop essentially computes the elements of Ki_inv_ki"
    "we recall that Ki_inv_Ki is a N-dimensional vector"
    """In the case where the point where we predict belong o group i 
    then only ni-1 weights are computed, but we still store them in a
    column of length ni"""

    for i in range(N):
        Ki = CovMatrix_scale(points_group[i], var, covtype)
        Ki_D_inv = mat_inverse(Ki + D_x[i])

        ki_x = CovMat1Mat2_scale(points_group[i], xscale, var, covtype)
        Ki_inv_ki[i] = np.matrix(np.zeros((gpsize[i], q), dtype='float64'))

        "k-leave one out Simple Kriging"
        for j in range(q):
            if group_x[j] == i:
                "x is in group i, in position position_x"
                position_x = position_in_group_x[j]

                "Then we need to transform Ki and Ki_x by removing rows / columns"
                Ki_x = Ki
                Ki_x = np.delete(Ki_x, position_x, axis=0)
                Ki_x = np.delete(Ki_x, position_x, axis=1)
                Ki_x = np.matrix(Ki_x)

                Di_x = D_x[i]
                Di_x = np.delete(Di_x, position_x, axis=0)
                Di_x = np.delete(Di_x, position_x, axis=1)
                Di_x = np.matrix(Di_x)

                "Since ki_x is a column vector, we only need to remove a row"
                ki_xx = ki_x[:, j].reshape(-1, 1)
                ki_xx = np.delete(ki_xx, position_x, axis=0)

                points_response_x = points_response_noise[i].reshape(1, -1)
                points_response_x = np.delete(points_response_x, position_x, axis=1)
                points_response_x = np.matrix(points_response_x)

                "new kriging weights in group i"
                tmp_Ki_inv_ki = mat_inverse(Ki_x + Di_x) * ki_xx

                "new prediction"
                Y_hat[i, j] = points_response_x * tmp_Ki_inv_ki
                k_Mx[i, j] = ki_xx.T * tmp_Ki_inv_ki
                K_Mx[j][i, i] = nuggetfactor * k_Mx[i, j]

                """In order to be able to store these kriging weights,
                we need to add an extra element equal to zero"""
                tmp_Ki_inv_ki = np.insert(tmp_Ki_inv_ki, gpsize[i] - 1, values=0, axis=0)
                Ki_inv_ki[i][:, j] = tmp_Ki_inv_ki
            else:
                "x is not in group i, everything is dealt with normally then"
                tmp_Ki_inv_ki = Ki_D_inv * ki_x[:, j]
                Y_hat[i, j] = points_response_noise[i] * tmp_Ki_inv_ki
                k_Mx[i, j] = ki_x[:, j].T * tmp_Ki_inv_ki
                K_Mx[j][i, i] = nuggetfactor * k_Mx[i, j]
                Ki_inv_ki[i][:, j] = tmp_Ki_inv_ki

    "(4) Second loop of the off-diagonal elements"
    "off-diagonal elements of k_Mx"
    for i in range(N):
        for j in range(i):
            "Computing the covariance between the points from group i, and the point from group j"
            Kij = CovMat1Mat2_scale(points_group[i], points_group[j], var, covtype)

            "If the point where we predict belongs to group i or j, then some extra work is needed"
            for k in range(q):
                if group_x[k] == i:
                    newKij = np.delete(Kij, position_in_group_x[k], axis=0)
                    Ki_inv_ki_x = Ki_inv_ki[i][:, k]
                    "Remove the last element we added before"
                    Ki_inv_ki_x = np.delete(Ki_inv_ki_x, gpsize[i] - 1, axis=0)

                    K_Mx[k][i, j] = Ki_inv_ki_x.T * newKij * Ki_inv_ki[j][:, k]
                    K_Mx[k][j, i] = K_Mx[k][i, j]
                elif group_x[k] == j:
                    newKij = np.delete(Kij, position_in_group_x[k], axis=1)
                    Kj_inv_kj_x = Ki_inv_ki[j][:, k]
                    Kj_inv_kj_x = np.delete(Kj_inv_kj_x, gpsize[j] - 1, axis=0)

                    K_Mx[k][i, j] = Ki_inv_ki[i][:, k].T * newKij * Kj_inv_kj_x
                    K_Mx[k][j, i] = K_Mx[k][i, j]
                else:
                    "case where the prediction point belongs neither to group i or j"
                    K_Mx[k][i, j] = Ki_inv_ki[i][:, k].T * Kij * Ki_inv_ki[j][:, k]
                    K_Mx[k][j, i] = K_Mx[k][i, j]

    "(5) Conclusion of the algorithm"

    for i in range(q):
        K_Mx_inv = mat_inverse(K_Mx[i])
        k_Mxi = k_Mx[:, i]

        "weights is coefficients of M_A, specifically, alpha.T"
        weights[:, i] = K_Mx_inv * k_Mxi

        predmean[i] = k_Mxi.T * K_Mx_inv * Y_hat[:, i]
        predsd2[i] = max(0, var - k_Mxi.T * K_Mx_inv * k_Mxi)

        "The LOOerror is taken as the difference between the prediction and the actual value"
        LOOerror[i] = predmean[i] - response_x[i]

    "(6) returns the final result"

    Result = {}

    Result['m_A'] = predmean
    Result['v_A'] = predsd2
    Result['K_Mx'] = K_Mx
    Result['k_Mx'] = k_Mx
    Result['m'] = Y_hat
    Result['LOOerror'] = LOOerror

    return Result
