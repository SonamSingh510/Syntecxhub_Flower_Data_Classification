# 🌸 Flower Data Classification

The objective was to build a classifier that uses physical measurements (Sepal length/width and Petal length/width) to identify flower species. The project focuses on the end-to-end ML lifecycle: from Exploratory Data Analysis (EDA) to model comparison and interactive prediction.

## 🛠️ Key Features

* **Deep EDA:** Used `Seaborn` pair plots to visualize feature distributions and species clustering.
* **Model Benchmarking:** Compared **Logistic Regression** and **Decision Tree** algorithms to find the most robust solution.
* **Error Analysis:** Implemented a **Confusion Matrix** to visualize true positives and misclassifications.
* **Interactive CLI:** Built a script that accepts real-time user input for on-the-spot flower classification.

## 💻 Tech Stack

* **Language:** Python 3.10
* **ML Libraries:** `scikit-learn`, `numpy`
* **Data Handling:** `pandas`
* **Visualization:** `matplotlib`, `seaborn`

## 📂 Project Structure

```text
├── flowerdata.py   # Main ML pipeline & comparison script
├── README.md                  # Project documentation
└── requirements.txt           # List of necessary Python libraries
```

## ⚙️ How to Run

1. **Clone the repo:** `git clone https://github.com/SonamSingh510/flowerdata.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Execute:** `python flowerdata.py`
