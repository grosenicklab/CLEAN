# Copyright (C) Responsive Capital Management - All Rights Reserved.
# Unauthorized copying of any code in this directory, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Logan Grosenick <logan@responsive.ai>, 2019.

import numpy as np
def apply_transform(transformer):
    transformer.fit(X_train_sel)
    X_train_sel = transformer.transform(X_train_sel)  
    X_test_sel = transformer.transform(X_test_sel)  
