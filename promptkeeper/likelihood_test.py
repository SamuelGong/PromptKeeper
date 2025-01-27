import numpy as np
from scipy.stats import norm
from scipy.optimize import root


def derive_mll_given_likelihood_ratio(likelihood_ratio, fitted_params):
    zero_mean, zero_std, other_mean, other_std = fitted_params
    A = other_std ** 2 - zero_std ** 2
    B = 2 * (zero_std ** 2 * other_mean - zero_mean * other_std ** 2)
    C = (zero_mean ** 2 * other_std ** 2
         - zero_std ** 2 * other_mean ** 2
         - 2 * zero_std ** 2 * other_std ** 2
         * np.log(other_std * likelihood_ratio / zero_std))

    Delta = B ** 2 - 4 * A * C
    if Delta >= 0:
        mll_1 = (- B - np.sqrt(Delta)) / (2 * A)
        mll_2 = (- B + np.sqrt(Delta)) / (2 * A)
        return min(mll_1, mll_2), max(mll_1, mll_2)
    else:
        return None, None


def _get_likelihood_ratio_given_significance_for_norm(likelihood_ratio,
                                                      fitted_params, significance):
    zero_mean, zero_std, other_mean, other_std = fitted_params
    mll_1, mll_2 = derive_mll_given_likelihood_ratio(likelihood_ratio, fitted_params)
    if mll_1 is not None:
        if zero_std > other_std:
            # the second derivative at the extremum point is less than zero.
            # i.e.，the extremum point is a maximal one
            rejection_area = (norm.cdf(mll_1, loc=other_mean, scale=other_std)
                              + (1 - norm.cdf(mll_2, loc=other_mean, scale=other_std)))
        else:
            # the second derivative at the extremum point is larger than zero.
            # i.e.，the extremum point is a minimal one
            rejection_area = (norm.cdf(mll_2, loc=other_mean, scale=other_std)
                              - norm.cdf(mll_1, loc=other_mean, scale=other_std))
        # TODO: we have ignored the situation when they are equal
    else:
        if zero_std > other_std:
            rejection_area = 1.0
        else:
            rejection_area = 0.0
        # TODO: we have ignored the situation when they are equal

    print(rejection_area, likelihood_ratio, significance)
    equation = rejection_area - significance
    return equation


def get_initial_guess_for_norm(fitted_params, initial_guess_multiplier=None):
    zero_mean, zero_std, other_mean, other_std = fitted_params
    # the mll correspond to the minimum possible likelihood ratio
    mll_for_initial_guess = ((other_std ** 2 * zero_mean - zero_std ** 2 * other_mean)
                     /(other_std ** 2 - zero_std ** 2))
    other_prob = norm.pdf(mll_for_initial_guess, loc=other_mean, scale=other_std)
    zero_prob = norm.pdf(mll_for_initial_guess, loc=zero_mean, scale=zero_std)

    initial_guess = other_prob / zero_prob
    if zero_std > other_std:
        # the second derivative at the extremum point is less than zero.
        # i.e.，the extremum point is a maximal one
        initial_guess /= 10  # important for stability
        if initial_guess_multiplier is not None:
            initial_guess *= initial_guess_multiplier  # IMPORTANT
    else:
        # the second derivative at the extremum point is larger than zero.
        # i.e.，the extremum point is a minimal one
        initial_guess += 1e-8  # important for stability
        if initial_guess_multiplier is not None:
            initial_guess *= initial_guess_multiplier  # IMPORTANT
    # TODO: we have ignored the situation when they are equal

    return initial_guess


def get_likelihood_ratio_given_significance(dist, fitted_params,
                                            significance):
    if dist == 'norm':
        # try again if possible with another initial guess
        zero_mean, zero_std, other_mean, other_std = fitted_params

        if zero_std <= other_std:
            print('A')
            for initial_guess_multiplier in [None, 10, 100, 1000]:
                initial_guess = get_initial_guess_for_norm(
                    fitted_params,
                    initial_guess_multiplier=initial_guess_multiplier
                )
                solution = root(
                    _get_likelihood_ratio_given_significance_for_norm,
                    method='hybr',
                    x0=initial_guess,
                    args=(fitted_params, significance)
                )
                if solution.success:
                    likelihood_ratio = solution.x[0]
                    return likelihood_ratio
                else:
                    print(f"Initial guess multiplier {initial_guess_multiplier} not suitable.")
            # if none succeeds
            raise ValueError("Solution for likelihood ratio "
                             "not found due to", solution.message)
        else:
            print('B')
            for initial_guess_multiplier in np.arange(1, 10 + 0.5, 0.5):
                initial_guess = get_initial_guess_for_norm(
                    fitted_params,
                    initial_guess_multiplier=initial_guess_multiplier
                )
                solution = root(
                    _get_likelihood_ratio_given_significance_for_norm,
                    method='hybr',
                    x0=initial_guess,
                    args=(fitted_params, significance)
                )
                if solution.success:
                    likelihood_ratio = solution.x[0]
                    return likelihood_ratio
                else:
                    print(f"Initial guess multiplier {initial_guess_multiplier} not suitable.")
            # if none succeeds
            raise ValueError("Solution for likelihood ratio "
                             "not found due to", solution.message)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    dist = "norm"
    # likelihood_ratio = 0.05
    zero_mean = -0.89
    zero_std = np.sqrt(0.047)
    other_mean = -0.278
    other_std = np.sqrt(0.053)
    significance = 0.2
    fitted_params = [zero_mean, zero_std, other_mean, other_std]

    lr = get_likelihood_ratio_given_significance(dist, fitted_params, significance)
    print("Solution for likelihood ratio:", lr)
