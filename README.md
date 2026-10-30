# Machine Learning Projects

End-to-end machine learning projects in **PyTorch** and **scikit-learn**. Each one follows the full pipeline: data cleaning, exploratory analysis, a baseline, a neural network, evaluation, and saving the model.

I designed these projects and wrote the reference solutions as the final projects of a machine learning training program that I teach. The notebooks are written in **French** and run as-is on Google Colab or locally.

| # | Project | Task | Techniques | Result (test set) |
|---|---|---|---|---|
| 01 | [Heart disease risk](01-heart-disease) | Binary classification | Missing-value imputation, encoding, logistic-regression baseline, MLP with BCE loss, ROC analysis | Accuracy **93.0 %**, F1 **0.91** |
| 02 | [California housing prices](02-california-housing) | Regression | Full EDA, outlier handling, MLP with BatchNorm, Dropout, early stopping, LR scheduler | R² **0.76** |
| 03 | [Iris on dirty data](03-iris-data-quality) | Multi-class classification | Dataset deliberately degraded with missing values, outliers and noisy features (generator included); SGD vs Adam vs RMSprop comparison | Accuracy 72 % on the degraded data. The point is the cleaning, not the score. |
| 04 | [Web-service quality (QWS)](04-web-service-qos) | Clustering → classification | K-Means with elbow/silhouette to create Bronze/Silver/Gold labels, then a multi-class MLP, feature importance | Accuracy **97.9 %** |
| 05 | [SMS spam detection](05-sms-spam) | NLP binary classification | TF-IDF vectorization, MLP classifier, precision/recall trade-off | Accuracy **98.4 %** |
| 06 | [Handwritten digits (MNIST)](06-mnist) | Image classification (10 classes) | MLP with BatchNorm and Dropout, training curves, confusion matrix, error analysis | Accuracy **98.7 %** |

## Run

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn
jupyter notebook
```

Most datasets are included or downloaded automatically: MNIST via `torchvision`, SMS Spam from the UCI repository.
The QWS dataset is **not redistributed**. Download `qws2.txt` from the [QWS dataset page](https://qwsdata.github.io/), put it in `04-web-service-qos/`, and cite the authors (Al-Masri & Mahmoud).

## Data sources

* California Housing: StatLib / scikit-learn
* SMS Spam Collection: Almeida & Hidalgo, UCI Machine Learning Repository (CC BY 4.0)
* MNIST: LeCun, Cortes & Burges
* QWS: Al-Masri & Mahmoud, *QoS-based Discovery and Ranking of Web Services* (ICCCN 2007)
* Iris: Fisher (1936), via scikit-learn, degraded by `generate_iris_preprocessing_dataset.py`

## Author

**Abdelhak Samir Douli**, PhD student working on machine learning for fault diagnosis and control of renewable energy systems (Université de Lille / Université Sétif 1).
See my research work on my [GitHub profile](https://github.com/abdelhaksamirdouli).
