# Data pulled from 01_fetch_chembl.py will be reused
# A number of compounds have not been included, as only one assay type been used to create the model
# Therefore, an other assay type can be used to validate the model
# While the 'cut-off' value is rather vague, we can optimize for best hit-ratio
# This is likely not the ideal way to validate this model, but it should work to give us an idea

import pandas
import chemprop

# Start with fetching the used data
# For this step, we can reuse the fetch_chembl code



