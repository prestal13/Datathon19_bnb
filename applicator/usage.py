#!/usr/bin/env python3

import os
import pandas as pd


from applicator import StaticParams, TemporalParams, TemporalState, PickleLoader, \
        OutputSeries, Applicator


def main():
    data_static = pd.read_csv('raw_data/Static.csv', ';')
    data_repay  = pd.read_csv('raw_data/Repayments.csv', ';')
    data_join   = pd.merge(data_static, data_repay, on='CONTRACT_ID')

    fixed_data = data_join[data_join['CONTRACT_ID'] == data_join['CONTRACT_ID'][1]]

    term = fixed_data['TERM'][0]
    contract_sum            = fixed_data['CONTRACT_SUM'][0]
    gender                  = fixed_data['GENDER'][0]
    age                     = fixed_data['AGE'][0]
    loan_to_income          = fixed_data['LOAN_TO_INCOME'][0]
    payment_to_income       = fixed_data['PAYMENT_TO_INCOME'][0]
    downpayment             = fixed_data['DOWNPAYMENT'][0]
    car_category            = fixed_data['CAR_CATEGORY'][0]
    grace_period            = fixed_data['GRACE_PERIOD'][0]
    rate_change_after_grace = fixed_data['RATE_CHANGE_AFTER_GRACE'][0]
    scheduled_list          = fixed_data['REPAYMENT_SCHEDULED'].tolist()

    # Test Case A.
    static_params = StaticParams(term, contract_sum,
            gender, age, loan_to_income,
            payment_to_income, downpayment,
            car_category, grace_period,
            rate_change_after_grace, scheduled_list)

    known_list = (fixed_data['REPAYMENT_ACTUAL'] / contract_sum).tolist()

    temporal_params = TemporalParams(known_list, static_params)

    loader = PickleLoader()
    clf, est = loader.load('./pickle_data/rf_clf.pickle', './pickle_data/rf_est.pickle')

    applicator = Applicator(clf, est, 'km')

    new_series = applicator.generate_new_series(static_params, temporal_params)
    complete_series = applicator.complete_source_series(static_params, temporal_params)
    by_one_series = applicator.by_one_series(static_params, temporal_params)
    cluster = applicator.get_cluster(static_params)

    print("Case A")
    print(cluster)
    print(complete_series.percent_series)
    print(by_one_series.percent_series)
    print(new_series.percent_series)

    # Test Case B.
    static_params = StaticParams(term, contract_sum,
            gender, age, loan_to_income,
            payment_to_income, downpayment,
            car_category, grace_period,
            rate_change_after_grace, None)
    temporal_params = TemporalParams(known_list, static_params)

    new_series = applicator.generate_new_series(static_params, temporal_params)
    complete_series = applicator.complete_source_series(static_params, temporal_params)
    by_one_series = applicator.by_one_series(static_params, temporal_params)
    cluster = applicator.get_cluster(static_params)

    print("Case B")
    print(cluster)
    print(complete_series.percent_series)
    print(by_one_series.percent_series)
    print(new_series.percent_series)


if __name__ == '__main__':
    main()

