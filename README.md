# Predictors

### Overview

The model uses LightGBM to predict household expenditure patterns based on demographic and socioeconomic features. It processes both household-level and person-level data to generate predictions.

### Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
import pandas as pd
from inference import mpce_predict

# Load your data
household_data = pd.read_csv('path/to/household_data.csv')
person_data = pd.read_csv('path/to/person_data.csv')

# Get predictions
predictions = mpce_predict(household_data, person_data)
```



### Features Used

The model aggregates person-level data to create household features including:
- Count and mean age of adults, children, and elders
- Gender distribution and ratios
- Dependency ratio
- Education levels
- Internet usage
- Meal consumption patterns


