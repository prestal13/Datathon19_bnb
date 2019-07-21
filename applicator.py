import numpy as np
import pandas as pd
import sklearn as sk
import pickle


class StaticParams():
    def __init__(self, term: int, contract_sum: float,
            gender: str, age: int, loan_to_income: float,
            payment_to_income: float, downpayment: float,
            car_category: int, grace_period: int,
            rate_change_after_grace: float,
            scheduled_list : list):

        ''' Represents static input.

        Args:
            term - # of month in contract, e.g. 12, 60, 84, 120, 18, 6
            contract sum - # amount of money in contract (in DTK)
            gender - can be 'M' or 'F' (for male and female)
            age - how old at the begin of credit
            loan_to_income -- ratio of loan to income
            payment_to_income -- ratio of payment to income
            downpayment -- what in dataset was a downpayment
            car_category -- should be one of [1, 2, 3, 4, 5]
            grace_period -- length of grace period
            rate_change_after_grace -- how rate changed after grace
            scheduled_list : scheduled payments for the time series if list (can be None)
        '''

        self.term = term
        self.contract_sum = contract_sum
        self.gender = gender
        assert self.gender in ['M', 'F']
        if self.gender == 'M' :
            self.idx_gender = 1
        elif self.gender == 'F' :
            self.idx_gender = 0
        else:
            raise ValueError("Unknown gender!")
        self.age = age
        self.loan_to_income = loan_to_income
        self.payment_to_income = payment_to_income
        self.downpayment = downpayment
        assert sefl.car_category in [1, 2, 3, 4, 5]
        self.car_category = car_category
        self.grace_period = grace_period
        self.rate_change_after_grace = rate_change_after_grace

        self.before_grace = self.payment_to_income / self.loan_to_income

        self.ratio_10 = int(self.loan_to_income >= 10)
        self.ratio_20 = int(self.loan_to_income >= 20)
        self.ratio_30 = int(self.loan_to_income >= 30)
        self.ratio_40 = int(self.loan_to_income >= 40)

        if scheduled_list is not None:
            self.scheduled_list = np.array(scheduled_list)
        else:
            self.scheduled_list = np.zeros(shape=self.term)
            self.scheduled_list += self.before_grace / 100. * self.contract_sum

            for idx in range(self.grace_period, self.term):
                self.scheduled_list[idx] += self.rate_change_after_grace / 100. * self.contract_sum


class TemporalParams():
    def __init__(self, scheduled_list : list, actual_known_list : list, static_params: StaticParams):
        ''' Contains Temporal Parameters
        Args:
            actual_known_list : known actual payments (and np.nans for unknown)
        '''
        self.static_params = static_params
        self.periods = list(range(1, static_params.term + 1))
        while len(actual_known_list) < static_params.term:
            actual_known_list.append(np.nan)
        actual_known_list = actual_known_list[:self.static_params.term]
        self.actual_known_list = np.array(actual_known_list)

        self.scheduled_percent = self.scheduled_list / static_params.contract_sum
        self.actual_percent = self.actual_known_list / static_params.contract_sum


class TemporalState():

    ''' Temporal State for generation. '''

    def __init__(self):
        self.accumulated_pay_actual = 0
        self.accumulated_percent_actual = 0
        self.accumulated_pay_scheduled = 0
        self.accumulated_percent_scheduled = 0

        self.avg_pay_actual = 0
        self.avg_percent_actual = 0
        self.avg_pay_scheduled = 0
        sefl.avg_percent_scheduled = 0

        self.last_actual_pay = 0
        self.prelast_actual_pay = 0
        self.prepre_actual_pay = 0

        self.last_actual_percent = 0
        self.prelast_actual_percent = 0
        self.prepre_actual_percent = 0

        self.grace_on = None
        self.is_grace_constant = 1

        self.pay_scheduled = None
        self.percent_scheduled = None


class PickleLoader():

    def load(clf_path, est_path):
        clf = pickle.load(open(clf_path, 'rb'))
        est = pickle.load(open(est_path, 'rb'))
        return clf, est


class OutputSeries:

    def __init__(self):
        self.percentage_series = list()
        self.pay_series = list()

        # Transformation is unknown.
        self.transformed_series = list()


class Applicator():
    def __init__(self, dict_clf, est, way) :
        '''
        Applies learned algorithms.
        Args:
            way of clustering,  should be one of: [sdtw_km, dba_km, km, gak_km, ks]
            est : regressor
            dict_clf : dict from way to classifier
        '''
        self.clf = dict_clf[way]
        self.est = est

    def generate_new_series(
            self, static_params: StaticParams, temporal_params : TemporalParams):
        """ Makes New Series """
        return make_series(static_params, temporal_params, False, False)

    def complete_source_series(
            self, static_params : StaticParams, temporal_params : TemporalParams):
        """ Completes nans in source series"""
        return make_series(static_params, temporal_params, True, True)

    def by_one_series(
            self, static_params: StaticParams, temporal_params : TemporalParams):
        """ Generates series with one-predict shift """
        return make_series(static_params, temporal_params, True, False)

    def get_cluster(self, static_params: StaticParams):
        """ Returns the cluster """
        features = [static_params.term, static_params.contrac_sum,
                static_params.idx_gender,
                static_params.age, static_params.loan_to_income,
                static_params.payment_to_income, static_params.downpayment,
                static_params.car_category, static_params.grace_period,
                static_params.rate_change_after_grace]
        return self.clf.predict(np.array([features]))

    # The following methods are just PRIVATE (though not underscored _private_method due lack of time)

    def update_state(self, state: TemporalState, next_percent: float, static_params: StaticParams, id_period: int):
        contract_sum = static_params.contract_sum
        next_value = next_percent / 100 * contract_sum

        if id_period > static_params.grace_period:
            state.grace_on = 0

        if abs(next_value - state.last_actual_pay) > 1e-3 and state.grace_on == 1:
            self.is_grace_constant = 0

        state.accumulated_pay_actual += next_value
        state.accumulated_percent_actual += next_percent

        if id_period < len(static_params.scheduled_list) :
            state.accumulated_pay_scheduled += static_params.scheduled_list[id_period]
            state.pay_scheduled = static_params.scheduled_list[id_period]
            state.percent_scheduled = state.pay_scheduled / contract_sum * 100
        state.accumulated_percent_scheduled = (state.accumulated_pay_scheduled
                / contract_sum * 100)

        state.avg_pay_actual = state.accumulated_pay_actual / id_period
        state.avg_percent_actual = state.avg_pay_actual / contract_sum * 100

        state.avg_pay_scheduled = state.accumulated_pay_scheduled / id_period
        state.avg_percent_scheduled = state.avg_pay_scheduled / contract_sum * 100

        state.prepre_actual_pay = state.prelast_actual_pay
        state.prelast_actual_pay = state.last_actual_pay
        state.last_actual_pay = next_value

        state.last_actual_percent = state.last_actual_pay / contract_sum * 100
        state.prelast_actual_percent = state.prelast_actual_pay / contract_sum * 100
        state.prepre_actual_percent= state.prepre_actual_pay / contract_sum * 100

    def update_output(self, output: OutputSeries, next_percent: float, static_params: StaticParams):
        contract_sum = static_params.contract_sum
        next_value = next_percent / 100 * contract_sum
        output.percentage_series.append(next_percent)
        output.pay_series.append(next_value)

    def init_state(self, state: TemporalState, static_params: StaticParams):
        state.grace_on = static_params.grace_period
        if state.grace_on > 0 : state.grace_on = 1
        state.pay_scheduled = static_params.scheduled_list[0]
        state.percent_scheduled = state.pay_scheduled / static_params.contract_sum
        self.init_state(state, static_params)

    def make_series(self, static_params : StaticParams, temporal_params : TemporalParams
            real_state_update: bool, real_output_update: bool):
        output = OutputSeries()
        state = TemporalState()
        self.init_state(state, static_params)
        for id_period in range(1, sttic_params.term + 1):
            features = [static_params.term, static_params.contract_sum, static_params.gender,
                    static_params.age, static_params.loan_to_income, static_params.payment_to_income,
                    static_params.downpayment, static_params.car_category, static_params.grace_period,
                    static_params.rate_change_after_grace, id_period, state.pay_scheduled,
                    state.accumulated_pay_scheduled, state.accumulated_pay_actual,
                    state.avg_pay_scheduled, state.avg_pay_actual, state.last_actual_pay,
                    state.prelast_actual_pay, state.prepre_actual_pay,
                    static_params.before_grace, state.percent_scheduled,
                    state.accumulated_percent_actual, state.accumulated_percent_scheduled,
                    state.avg_percent_actual, state.avg_percent_scheduled,
                    state.grace_on, state.last_actual_percent,
                    state.prelast_actual_percent, state.prepre_actual_percent,
                    static_params.ratio_10, state_params.ratio_20, state_params.ratio_30,
                    state_params.ratio_40, state_params.is_grace_constant]
            estimate = self.est.predict(np.array([features]))[0]
            if estimate  + state.avg_percent_actual > 100:
                estimate = 100. - state.avg_percent_actual - 1e-8
                if estimate < 0 : estimate = 0
            real  = temporal_params.actual_known_list[id_period -1]
            if real is np.nan:
                real = estimate
            if real_state_update:
                self.update_state(state, real, static_params)
            else:
                self.update_state(state, estimate, static_params)
            if real_output_update:
                self.update_output(state, real, static_params)
            else:
                self.update_output(state, estimate, static_params)
        self.transform_output(output, static_params, temporal_params)
        return output

    def transform_output(self, output: OutputSeries, static_params: StaticParams, temporal_params: TemporalParams):
        pass


