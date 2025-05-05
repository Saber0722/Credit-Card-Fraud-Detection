# Credit Card Fraud Detection

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. 
## Table of Contents

* [Repository Structure](#repository-structure)
* [Setup Instructions](#setup-instructions)
* [Usage](#usage)
* [Streamlit Application](#streamlit-application)
* [Contributing](#contributing)
* [License](#license)

## Repository Structure

```
Credit_Card_Fraud_Detection/
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for EDA, data preprocessing and model development
├── outputs/             # Generated outputs such as figures and model predictions
├── src/                 # Source code for the streamlit app
├── requirements.txt     # List of required Python packages
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Saber0722/Credit_Card_Fraud_Detection.git
   cd Credit_Card_Fraud_Detection
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation:**

   Ensure that the necessary datasets are placed in the `data/` directory. If data preprocessing is required, utilize the scripts available in the `src/` directory.

2. **Exploratory Data Analysis:**

   Navigate to the `notebooks/` directory and open the relevant Jupyter notebooks to perform EDA and understand data distributions and relationships.

3. **Model Development:**

   Use the notebooks or scripts in the `src/` directory to train and evaluate your models. Adjust parameters as needed to optimize performance.

4. **Results Visualization:**

   Generated outputs, including plots and model predictions, will be saved in the `outputs/` directory for review and analysis.

## Streamlit Application

For an interactive exploration of the analysis and results, access the Streamlit application:

👉 [Launch the Streamlit App](https://huggingface.co/spaces/Saber-0722/Credit-Card-Fraud-Detection)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request detailing your changes.

## License

This project is licensed under the [MIT License](LICENSE).
