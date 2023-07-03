# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

class PredictionModel(object):
    '''
    Model object instantiated at beginning of Prediciton Pipeline in order to store model 
    settings, parameters, and results as the pipeline progresses.
    
    Attrs:
        pipeline_version
        date_run
        random_seed
        data_object: user_ids, transaction_dates, feature_ids for loaded data.
        stats_object: feature engineering parameters: (stats, xcorr, window_len, k_ahead)
        validation_object: indices and CV setup
        screening: method used, variables kept, variable importance stats
        embedding: method, parameters, goodness-of-fit
        event_detection: method, parameters, coefficients, model fit
        regression: method, parameters, coefficients, model fit
        prediction: method, goodness-of-fit
        visualization: plots made and paths to them
    
    '''
    def __init__(self):
        pass
