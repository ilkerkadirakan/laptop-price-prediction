# Laptop Features Prediction App

This project is a Streamlit web application that predicts the price of laptops based on various features such as brand, processor, RAM, and more. The prediction model is trained using a dataset of laptops and is capable of preprocessing the input features before making predictions.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Features

- Users can select laptop specifications such as brand, processor, RAM, SSD, and more.
- The app preprocesses the input features to handle outliers and missing values.
- One-hot encoding and label encoding techniques are used for categorical variables.
- Predicts the price of laptops based on the input features using a pre-trained model.
- Provides an intuitive and interactive user interface with Streamlit.

## Installation

To run the application locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/ilkerkadirakan/laptop-price-predictor.git
    cd laptop-price-predictor
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run streamlit_app/laptopPrice.py
    ```

## Usage

Once the app is running, follow these steps:

1. Select the desired laptop specifications from the available options (brand, processor, RAM, etc.).
2. Click on the **Predict** button to get the predicted price.
3. The app will display the predicted price in Indian Rupees.

## Technologies Used

- **Python**
- **Streamlit** for the web interface.
- **Pandas** and **NumPy** for data manipulation.
- **Seaborn** and **Matplotlib** for data visualization.
- **Scikit-learn** for machine learning model and preprocessing.
- **Joblib** for model persistence.

## Dataset

The dataset used in this project includes laptop specifications and prices. The dataset is preprocessed before training the model. The preprocessed dataset is stored as `dataset.pkl` at streamlit_app folder. 
TThe raw dataset is `laptopPrice.csv`. It is in the main folder.

## Contributing

Contributions are welcome! If you would like to improve the application, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
