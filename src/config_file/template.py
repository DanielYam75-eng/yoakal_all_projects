import pandas as pd

allowed_templates = pd.DataFrame(
        [["monim-no-re-1", 'BI', True],
         ["invoices-to-po-1", 'BW', False],
         ["po-1", 'BW', False],
         ["monim-by-posting-date-1", 'BW', True],
         ["po-foreign-currency-1", 'BW', False],
         ["po-foreign-currency-2", 'BW', True],
         ["invoices-to-po-2", 'BW', True],
         ["invoices-to-po-foreign-currency-1", 'BW', True],
         ["po-2", 'BW', True],
         ["ZH-foreign-currency-1", 'BI', True],
         ["simulator-data-1", 'BW', True],
         ["simulator-time-axis-1", 'BW', True],
         ["NN-data-1", 'BI', True],
         ["ZH-1", 'BI', True],
         ["monim-by-posting-date-foreign-currency-1", 'BI', True],
         ["NN-data-2", 'BI', True],
         ["invoice-history-1", 'BI', True],
         ],
        columns=['template', 'source', 'allowed'])

