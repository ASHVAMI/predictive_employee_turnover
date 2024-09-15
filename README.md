# Predictive Employee Turnover Analysis

This project uses machine learning to predict employee turnover based on hypothetical data.

## Project Structure:
- **data/**: Contains the dataset.
- **notebooks/**: Jupyter notebooks for EDA.
- **models/**: Saved machine learning models.
- **src/**: Source code for loading, training, and evaluating models.
- **scripts/**: Python scripts to train models and make predictions.
- **reports/**: Model performance reports.

## How to Run:

1. Install dependencies:
pip install -r requirements.txt

Train the models:
python scripts/train.py

Make predictions:
python scripts/predict.py

### 3. Dataset (Sample)
Here’s a sample for the `employee_data.csv` file (you can expand on this):

EmployeeID,Age,YearsAtCompany,MonthlyIncome,JobSatisfaction,Department,Turnover
1,28,3,5000,3,Sales,0
2,35,8,7000,2,HR,1
3,42,10,6000,4,Engineering,0
