# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from math import ceil

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from pybaselines import polynomial
from pybaselines.utils import ParameterWarning

from .conftest import AlgorithmTester, get_data


class TestPoly(AlgorithmTester):
    """Class for testing regular polynomial baseline."""

    func = polynomial.poly

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestModPoly(AlgorithmTester):
    """Class for testing ModPoly baseline."""

    func = polynomial.modpoly

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIModPoly(AlgorithmTester):
    """Class for testing IModPoly baseline."""

    func = polynomial.imodpoly

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestPenalizedPoly(AlgorithmTester):
    """Class for testing penalized_poly baseline."""

    func = polynomial.penalized_poly

    @pytest.mark.parametrize(
        'cost_function',
        (
            'asymmetric_truncated_quadratic',
            'symmetric_truncated_quadratic',
            'a_truncated_quadratic',  # test that 'a' and 's' work as well
            's_truncated_quadratic',
            'asymmetric_huber',
            'symmetric_huber',
            'asymmetric_indec',
            'symmetric_indec'
        )
    )
    def test_unchanged_data(self, data_fixture, cost_function):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('huber', 'p_huber', ''))
    def test_unknown_cost_function_prefix_fails(self, cost_function):
        """Ensures cost function with no prefix or a wrong prefix fails."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, cost_function=cost_function)

    def test_unknown_cost_function_fails(self):
        """Ensures than an unknown cost function fails."""
        with pytest.raises(KeyError):
            self._call_func(self.y, self.x, cost_function='a_hub')

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
    def test_weighting(self, weight_enum):
        """
        Tests that weighting is correctly applied by comparing to other algorithms.

        Weights were not included in the original penalized_poly method developed
        in [1]_, so need to ensure that their usage in pybaselines is correct.

        According to [1]_ (and independently verified), the penalized_poly function
        with the asymmetric truncated quadratic cost function, a threshold of 0, and
        an alpha_factor of 1 should be the same as the output of the ModPoly algorithm.

        Furthermore, the penalized_poly with any symmetric cost function and a threshold
        of infinity should equal to the output of a regular polynomial fit.

        Therefore, to ensure that weighting is correct for the penalized_poly, check
        both conditions.

        References
        ----------
        .. [1] Mazet, V., et al. Background removal from spectra by designing and
               minimising a non-quadratic cost function. Chemometrics and Intelligent
               Laboratory Systems, 2005, 76(2), 121–133.

        """
        if weight_enum == 0:
            # all weights = 1
            weights = None
        elif weight_enum == 1:
            # same as all weights = 1, but would cause issues if weights were
            # incorrectly multiplied
            weights = 2 * np.ones_like(self.y)
        elif weight_enum == 2:
            # binary mask, only fitting the first half of the data
            weights = np.ones_like(self.y)
            weights[self.x < 0.5 * (np.max(self.x) + np.min(self.x))] = 0
        else:
            # weight array where the two endpoints have weighting >> 1
            weights = np.ones_like(self.y)
            fraction = max(1, ceil(self.y.shape[0] * 0.1))
            weights[:fraction] = 100
            weights[-fraction:] = 100

        poly_order = 2
        tol = 1e-3

        poly_baseline = polynomial.poly(self.y, self.x, poly_order, weights=weights)[0]
        penalized_poly_1 = self._call_func(
            self.y, self.x, poly_order, cost_function='s_huber',
            threshold=1e10, weights=weights
        )[0]

        assert_array_almost_equal(poly_baseline, penalized_poly_1)

        modpoly_baseline = polynomial.modpoly(
            self.y, self.x, poly_order, tol=tol, weights=weights, use_original=True
        )[0]
        penalized_poly_2 = self._call_func(
            self.y, self.x, poly_order, cost_function='a_truncated_quadratic',
            threshold=0, weights=weights, alpha_factor=1, tol=tol
        )[0]

        assert_array_almost_equal(modpoly_baseline, penalized_poly_2)


class TestLoess(AlgorithmTester):
    """Class for testing LOESS baseline."""

    func = polynomial.loess

    @pytest.mark.parametrize('conserve_memory', (True, False))
    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_unchanged_data(self, data_fixture, use_threshold, conserve_memory):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(
            data_fixture, y, x, y, x, use_threshold=use_threshold, conserve_memory=conserve_memory
        )

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_x_ordering(self, use_threshold):
        """Ensures arrays are correctly sorted within the function."""
        reverse_x = self.x[::-1]
        reverse_y = self.y[::-1]

        if use_threshold:
            # test both True and False for use_original
            regular_inputs_result = self._call_func(
                self.y, self.x, use_threshold=use_threshold, use_original=False
            )[0]
            reverse_inputs_result = self._call_func(
                reverse_y, reverse_x, use_threshold=use_threshold, use_original=False
            )[0]

            assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

            regular_inputs_result = self._call_func(
                self.y, self.x, use_threshold=use_threshold, use_original=True
            )[0]
            reverse_inputs_result = self._call_func(
                reverse_y, reverse_x, use_threshold=use_threshold, use_original=True
            )[0]

            assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

        else:
            regular_inputs_result = self._call_func(
                self.y, self.x, use_threshold=use_threshold
            )[0]
            reverse_inputs_result = self._call_func(
                reverse_y, reverse_x, use_threshold=use_threshold
            )[0]

            assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

    @pytest.mark.parametrize('fraction', (-0.1, 1.1, 5))
    def test_wrong_fraction_fails(self, fraction):
        """Ensures a fraction value outside of (0, 1) raises an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, fraction)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3))
    def test_too_small_window_fails(self, poly_order):
        """Ensures a window smaller than poly_order + 1 raises an exception."""
        for num_points in range(poly_order + 1):
            with pytest.raises(ValueError):
                self._call_func(self.y, self.x, total_points=num_points, poly_order=poly_order)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3, 4))
    def test_high_polynomial_order_warns(self, poly_order):
        """Ensure a warning is emitted when using a polynomial order above 2."""
        if poly_order > 2:
            with pytest.warns(ParameterWarning):
                self._call_func(self.y, self.x, poly_order=poly_order)
        else:  # no warning should be emitted
            self._call_func(self.y, self.x, poly_order=poly_order)

    @pytest.mark.parametrize('conserve_memory', (True, False))
    def test_use_threshold_weights_reset(self, conserve_memory):
        """Ensures weights are reset to 1 after first iteration if use_threshold is True."""
        weights = np.arange(self.y.shape[0])
        one_weights = np.ones(self.y.shape[0])
        # will exit fitting loop before weights are reset on first loop
        _, params_first_iter = self._call_func(
            self.y, self.x, weights=weights, conserve_memory=conserve_memory,
            use_threshold=True, tol=1e10
        )
        assert_array_equal(weights, params_first_iter['weights'])

        # will exit fitting loop after first iteration but after reassigning weights
        _, params_second_iter = self._call_func(
            self.y, self.x, weights=weights, conserve_memory=conserve_memory,
            use_threshold=True, tol=-1, max_iter=1
        )
        # will exit fitting loop after second iteration
        _, params_third_iter = self._call_func(
            self.y, self.x, weights=weights, conserve_memory=conserve_memory,
            use_threshold=True, tol=-1, max_iter=2
        )

        assert_array_equal(one_weights, params_second_iter['weights'])
        assert_array_equal(one_weights, params_third_iter['weights'])

    @pytest.mark.parametrize('conserve_memory', (True, False))
    def test_comparison_to_statsmodels(self, conserve_memory):
        """
        Compares the output of loess to the output of statsmodels.lowess.

        The library statsmodels has a well-tested lowess implementation, so
        can compare the output of polynomial.loess to statsmodels to ensure
        that the pybaselines implementation is correct.

        Since pybaselines's loess is for calculating the baseline rather than
        smoothing, the following changes need to be made to match statsmodels:

        * statsmodels uses int(fraction * num_x) to determine the window size while
          pybaselines uses ceil(fraction * num_x), so need to specify total points
          instead of fraction.
        * statsmodels uses the input iterations as number of robust fittings, while
          pybaselines uses iterations as total number of fits (intial + robust fittings),
          so add 1.
        * statsmodels divides the residuals by 6 * median-absolute-value(residuals)
          when weighting residuals, while pybaselines divides by
          m-a-v * scale / 0.6744897501960817, so set scale to 4.0469385011764905 to
          get 6 and match statsmodels.
        * set symmetric weights to True.
        * set tol to -1 so that it goes through all iterations.

        The outputs from statsmodels were created using::

            from statsmodels.nonparametric.smoothers_lowess import lowess
            output = lowess(y, x, fraction, iterations).T[1]

        with statsmodels version 0.11.1.

        """
        num_x = 100
        fraction = 0.1
        total_points = int(num_x * fraction)
        # use set values since minimum numpy version is < 1.17
        # once min numpy version is >= 1.17, can use the following to create x and y:
        # random_generator = np.random.default_rng(0)
        # x = np.sort(random_generator.uniform(0, 10 * np.pi, num_x))
        # y = np.sin(x) + random_generator.normal(0, 0.3, num_x)
        x = np.array([
            0.08603252016391653, 0.4620121964065525, 0.5192309835763667, 0.8896887082266529,
            1.055121966462336, 1.2872212178963478, 1.634297372541321, 1.8399690787918987,
            2.6394198618724256, 2.8510910140793273, 3.3142319528623445, 3.610715163730215,
            3.9044742841509907, 4.244181877040637, 4.673559279481576, 4.721168691822169,
            5.518384072467553, 6.236471222720201, 6.267962530482449, 7.13636627051602,
            7.245838693824786, 7.520012836005392, 8.475599579967072, 9.383815765196083,
            9.415726735057563, 9.74653597091934, 10.111825144195999, 10.559428881564424,
            10.615794236187337, 11.2404676147093, 11.470274223088868, 12.053585858164322,
            12.218326883964336, 12.303073750654486, 12.709370879795477, 13.026797701038113,
            13.279110688808476, 13.358951239219458, 13.834856340638826, 14.147828458876802,
            14.452744299731492, 15.262967941601095, 15.626994858727365, 15.704690158414039,
            16.504492800213836, 16.628831939299175, 17.010505917383114, 17.078482794955864,
            17.95513917528035, 18.23167960456526, 18.670486089035595, 19.058024965549173,
            19.33289345358043, 19.484580245090765, 19.578001555572705, 19.76401547190605,
            20.010741575072398, 20.332058150420306, 20.43478083782305, 21.06828734519464,
            21.111341718376682, 21.536936621719185, 21.62819191149577, 22.341212411453007,
            22.666224692044207, 22.902685361628365, 22.917810368063478, 22.92280190156093,
            23.074481933138635, 23.80475373833605, 24.686185846584056, 24.72742260459408,
            25.017264774098162, 25.549638088547898, 25.63079532033328, 25.835635751138295,
            26.158287373224493, 26.93614976483979, 27.11756561187958, 27.132053628611267,
            27.535564205022077, 27.94408445848314, 27.958150040199374, 27.968793639376692,
            28.675062160988013, 29.127419326603402, 29.13588200930302, 29.281518641717938,
            29.343842478613347, 29.376166571460544, 29.597356268178427, 29.811944778549844,
            29.98934482165474, 30.071644682071696, 30.40829807484428, 30.5560353617595,
            30.813850946806603, 30.8251512961117, 31.261878704601713, 31.328274083621345
        ])
        y = np.array([
            -0.31643948442795794, 0.025294176131094415, 0.647017478746023, 1.0737896890796879,
            0.8206720006830988, 0.6377518395963726, 1.259897131794639, 0.5798730362015778,
            0.26741078101860366, 0.4727382381565013, -0.8468253620243291, -0.33619288971896866,
            -0.8654995424139683, -0.8595949238604239, -1.0219566797460828, -0.9393271400679253,
            -0.484142018099982, -0.27420802162579655, 0.4110724179260672, 0.9712041195512975,
            1.0738302517367937, 1.2942080292091869, 1.049213774810279, 0.29417434561985983,
            0.03172918534871679, -0.7442669944536149, -0.674770592505406, -1.1372411535695315,
            -1.3555687271138004, -0.8926273563547041, -1.059994748851263, -0.7995470629457199,
            -0.6539598102298176, -0.17974008110838563, 0.2501149777114302, 0.8410679994805202,
            0.6497348888787546, 1.0247191935429998, 1.3753305584133848, 1.3449928574031798,
            0.2410261733731892, 0.7990587775689757, 0.18276597078472304, 0.13040450954666363,
            -0.603565652310072, -0.6813004303259478, -0.8684109596218175, -1.0876855412294537,
            -1.350334701806057, -0.6119798501022928, -0.41923391123976983, 0.5310113576086335,
            0.37810702031798554, 0.6182397032157187, 0.410828769461069, 0.6390461696807992,
            0.9138160240699695, 0.5504920525015544, 1.0901014439969032, 0.7655100757551879,
            0.41489986158068176, -0.2807155537378368, 0.5089554617452788, -0.43223298454102355,
            -0.7839592638879793, -0.8612923921663656, -0.25467700223013867, -0.8175475502673442,
            -0.8575382314233819, -1.4167883371270387, 0.062340456353472684,
            -0.11906519352829137, 0.20486047721541026, 0.4192268707418591, 0.7527133733771633,
            0.7577128888236917, 1.038954288353462, 0.9274096611755345, 0.47334138166288053,
            1.2182396587983262, 0.09289095590039953, 0.2522979824352233, 0.24958510013390783,
            -0.012049601859495218, -0.20615208716141473, -0.8134120774853145,
            -0.8899122080685423, -0.6893786555970529, -1.019938175719467, -0.4753431711651658,
            -0.8640242335846221, -1.141749310181384, -1.5728985760169076, -1.3667810598063708,
            -0.5195189005816284, -0.7729527752716384, -0.6512918332086576, -0.06402954966432,
            -0.5382340490144448, -0.2632375970675537
        ])

        # outputs for iteration numbers from 0 to 3; test several iterations to ensure
        # weighting is correct
        statsmodels_outputs = {
            0: np.array([
                0.013949173555858115, 0.28450953728555345, 0.3235872359802879, 0.5654824453094635,
                0.6662153656116145, 0.785530170334511, 0.7506470516691395, 0.7066468110557571,
                0.22276406508876975, 0.057066368459986624, -0.2757522455273732, -0.4645446340925608,
                -0.6184296460333416, -0.7530368402016178, -0.7762575079967078, -0.7479540895316928,
                -0.40197196468678825, 0.15849673370691408, 0.18014175529804785, 0.7786503604739311,
                0.7896729843541994, 0.9213607881485303, 0.6117221206735213, 0.04238915039014286,
                0.011867565386790317, -0.3063316051973748, -0.6276671543213955, -0.9108500301537548,
                -0.9467508014226902, -0.9711786056041706, -0.8803946787628942, -0.5164455033883036,
                -0.33665555792536794, -0.2618288279042163, 0.1733249979935404, 0.5455634831830709,
                0.838881074083777, 0.8759154349258786, 0.9140394360646751, 0.8951455195299661,
                0.7947599612587716, 0.3821827601443787, 0.1512117349631967, 0.09301799334373848,
                -0.4965343116225852, -0.5767092464770565, -0.8284529823712354, -0.8333897829320087,
                -0.7961999810752187, -0.6688759468942046, -0.3043400334135521, 0.14139781590335151,
                0.40684222112537777, 0.49385608811180576, 0.5389406543303799, 0.633681218926093,
                0.7280361065264241, 0.7355179605022224, 0.7522593016658435, 0.4776899262477174,
                0.4520942101790715, 0.18673709685456896, 0.11439812612816182, -0.3744272253914193,
                -0.576991450149173, -0.7359246913680174, -0.7459521031614607, -0.7492664086445703,
                -0.8494030222551598, -0.6507205593587618, -0.2138275389166584, -0.1733819619391141,
                0.10855935515534255, 0.5649038887804986, 0.6211904489892349, 0.7611554486022242,
                0.7886392334805351, 0.7985530647362659, 0.6908910652230766, 0.6821904597960557,
                0.4422805681131109, 0.14751814767227792, 0.1382551476910366, 0.13141652432138526,
                -0.34179415758466986, -0.6686132853934371, -0.6748737335592727, -0.7823371105410748,
                -0.8143840666226682, -0.825596154630696, -1.0103668209266579, -1.1716761119973165,
                -1.0918251801003562, -1.1127686785924296, -0.8234247338878935, -0.6974499317651827,
                -0.5479975556623471, -0.5410878522967348, -0.29460927434031603, -0.2623235846170958
            ]),
            1: np.array([
                0.01346125764454193, 0.2692642398394642, 0.3061035256436901, 0.5332712747559366,
                0.6272399548736404, 0.7382692594538496, 0.7053387567902282, 0.6646796433786578,
                0.2323640059741657, 0.0732102907329617, -0.24208143437774257, -0.4344790148980033,
                -0.5944176245440754, -0.7386440408224565, -0.7732961317448773, -0.7468933864635734,
                -0.38864077929380175, 0.17376605338956985, 0.19575335549178777, 0.8004593165358889,
                0.8023627241804053, 0.9261251955073373, 0.5994439645248809, 0.05699878408812928,
                0.026778379675386994, -0.29169837149050243, -0.6026592561389593, -0.886440198981581,
                -0.9239793915493274, -0.9609694843339092, -0.8724345120389131, -0.5051580544040742,
                -0.3271780675113524, -0.2532521125138645, 0.17736628619186365, 0.551946966252646,
                0.837913844080486, 0.8785877440186318, 0.9393958885313628, 0.9188956576650167,
                0.7891363132059664, 0.3992718043014226, 0.1673797246537597, 0.10806424728859307,
                -0.5055534159783011, -0.5832066091639541, -0.8153769847974243, -0.8127875659982049,
                -0.7433780928306141, -0.6239392745287204, -0.2896337544128261, 0.12840721447996856,
                0.3924041730637772, 0.4844186732272839, 0.5310429518106545, 0.6279753242141739,
                0.7226446429902373, 0.7235517339825621, 0.7389667828447715, 0.48521139018302706,
                0.4591660736842548, 0.1957075636053834, 0.12443366754757555, -0.3866279810488643,
                -0.5977519177178532, -0.7518851267358441, -0.7616255982308762, -0.7648223824532712,
                -0.852675911156658, -0.5033274286373174, -0.1325593462844366, -0.0962563110744339,
                0.15101593034682986, 0.5636136209068374, 0.6200950996381333, 0.7600298538837871,
                0.7803653276682748, 0.7637445365292009, 0.6602361538819609, 0.6519445633670966,
                0.4237234581863734, 0.14624953450768802, 0.13757744072359518, 0.13118839776630803,
                -0.34175982603826516, -0.6719520156563255, -0.6783251218633565, -0.7877675374568502,
                -0.8205754954903806, -0.8316765329159181, -1.0041130525588309, -1.159110284262741,
                -1.0686241021543004, -1.0865289781985887, -0.8301676506585461, -0.7147444884905023,
                -0.5679481193833289, -0.5612212147658148, -0.3199989504639437, -0.28762875412878713
            ]),
            2: np.array([
                0.009632821670032841, 0.2625608627268248, 0.29896340674207683, 0.5232560468894007,
                0.6159304336895257, 0.7255391566074584, 0.6940970271682658, 0.6552320827338113,
                0.237006729183965, 0.07883508498982184, -0.23270924737933235, -0.4260654242619405,
                -0.5876180474849226, -0.7347025720277461, -0.7727618128545918, -0.7463492617833534,
                -0.3860056933483532, 0.1764240188412542, 0.1984457278388993, 0.8052960744507982,
                0.8065260859303459, 0.9288859197233666, 0.598362989014333, 0.059249987125926336,
                0.02912681367998666, -0.288564131875102, -0.5978613434024713, -0.8818149543395937,
                -0.9196946896640417, -0.9595401426176005, -0.8715597091873306, -0.5034937803298825,
                -0.32564302888101165, -0.2518252703610642, 0.17805050251945617, 0.5518797168689674,
                0.8378699835200704, 0.8791185038005354, 0.9449523497737788, 0.9251372451827247,
                0.7943227256588703, 0.4017342199761544, 0.16891164970863634, 0.10945830563044567,
                -0.5061528516633934, -0.5833965674913659, -0.8118362991601958, -0.8074048296684508,
                -0.7300984524620348, -0.6116517595281588, -0.28376513756146915, 0.12761290437844916,
                0.39103577604756684, 0.4833757160197309, 0.5300145368276965, 0.6268762416877547,
                0.7212053540247538, 0.7218010598361171, 0.7371642669851735, 0.4870840229211958,
                0.46112026509065546, 0.1986448394628882, 0.1274991389728427, -0.38643185770874733,
                -0.5993893750731988, -0.7513760032390301, -0.7608218644233942, -0.7639057159167137,
                -0.8408899914753452, -0.4414779953742434, -0.08149490003254517,
                -0.04942917198002014, 0.1729979788685168, 0.5640091213105332, 0.6203542740084158,
                0.7601187618196771, 0.7779580204561654, 0.7552101656857411, 0.6524288432383029,
                0.6442047380101015, 0.4177027955834176, 0.14424216043496363, 0.13577200849356333,
                0.12954031841513225, -0.34192682003191166, -0.672545738948012, -0.6789263071736452,
                -0.7884622058491338, -0.8212530125629236, -0.8322925612295267, -1.0029673626871698,
                -1.1572807374559146, -1.063641532624775, -1.081381934260377, -0.8303651121548383,
                -0.7167574250326637, -0.5708797172339388, -0.5642153599235998, -0.32535642053883235,
                -0.2932379764786771
            ]),
            3: np.array([
                0.007106331472409507, 0.259404964445307, 0.2957148378560415, 0.5194034836119471,
                0.6118138566010974, 0.7212223470091101, 0.6904598931449145, 0.6522574042502203,
                0.23875595379354567, 0.08088020256939216, -0.2294977703741831, -0.42317762492962857,
                -0.585268118753555, -0.733321795449864, -0.7725784968455178, -0.7461965061218885,
                -0.3853436325421424, 0.17713864942047153, 0.19916945554511944, 0.8065789237690831,
                0.8076768967841921, 0.9297017937123315, 0.598171636640104, 0.05989595535684206,
                0.02980300826380175, -0.28765441120426166, -0.5965191059465016, -0.8805269523851127,
                -0.9185021283550571, -0.9591390430363158, -0.8713205430033873, -0.5030803704283456,
                -0.3252804699266583, -0.25149179793725274, 0.17817845836393603, 0.5518385551773628,
                0.8377824917551533, 0.879484181963784, 0.9490546749668137, 0.9294451337103058,
                0.796727007213466, 0.4038053780544547, 0.17034305949441655, 0.11074127849421078,
                -0.5064491715658765, -0.5835580019853329, -0.8107264026696844, -0.8057097756794069,
                -0.725734631524126, -0.6075219493317787, -0.28170511937086584, 0.12756516512760932,
                0.3908011423820175, 0.48320473981083323, 0.5298383697871437, 0.6266453458882145,
                0.7208897297028004, 0.7213591144704607, 0.7367072332077464, 0.48790188023203035,
                0.46195031699963357, 0.19970635698784583, 0.1286095300811426, -0.38597719455088464,
                -0.5993971508213999, -0.7503800322752152, -0.75967609212089, -0.7627027202535341,
                -0.8337926944629892, -0.41910718421816456, -0.05942619548910799,
                -0.029552299992710543, 0.18159230858436773, 0.5642684398241131, 0.6205242852640489,
                0.7601508143598422, 0.777216093243909, 0.7524509125402858, 0.6499048318707901,
                0.6417024914571287, 0.4157578052225643, 0.1435747502329964, 0.1351686513197739,
                0.1289869115783057, -0.341974691255135, -0.6726754822580662, -0.6790581630565466,
                -0.7886240804504231, -0.8214152331220816, -0.832438899647899, -1.0025693005333978,
                -1.1566488665741346, -1.06210263238769, -1.0797871263797816, -0.8302873248767663,
                -0.7171663382187876, -0.5715459335050725, -0.5648992578038142, -0.3266929570620274,
                -0.29464213838044545
            ]),
        }
        for iterations in range(4):
            self._test_accuracy(
                statsmodels_outputs[iterations], y, x, conserve_memory=conserve_memory,
                total_points=total_points, max_iter=iterations + 1, tol=-1,
                scale=4.0469385011764905, symmetric_weights=True,
                assertion_kwargs={'err_msg': f'failed on iteration {iterations}'}
            )
