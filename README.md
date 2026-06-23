AI-Powered Fake News Detection System
A robust fake news detection system that combines multiple machine learning models, including DistilBERT, LighGBM, XGBoost, Random Forests and Logistic Regression, to analyze and classify news articles as real or fake.

Fake News Detection Python Flask

🌟 Features
Multiple Model Analysis: Combines predictions from:
DistilBERT (Transformer-based)
XGBoost
Random Forest
LightGBM
Logistic Regression
Interactive Web Interface: Modern, responsive UI with dark/light mode
Detailed Analysis: Shows confidence scores and reasoning for each model
Consensus Voting: Weighted voting system for final prediction
Visual Analytics: Interactive charts showing model confidence levels
Technologies Used
Backend
Python 3.8+: Core programming language
Flask: Web framework for building the application
Gunicorn: WSGI HTTP Server for production deployment
NLTK: Natural Language Processing toolkit for text preprocessing
Scikit-learn: Machine learning library for traditional ML models
Joblib: For model serialization and loading
Machine Learning & AI
Transformers (Hugging Face): For DistilBERT model implementation
PyTorch: Deep learning framework for BERT model
XGBoost: Gradient boosting framework
LightGBM: Light gradient boosting machine
Scikit-learn: For Random Forest and Logistic Regression
Frontend
HTML5/CSS3: For structure and styling
JavaScript: For interactive features
Chart.js: For data visualization
Font Awesome: For icons
Google Fonts: For typography
Development & Deployment
Git: Version control
GitHub: Code hosting
Render: Cloud platform for deployment
Conda: Environment management
🚀 Quick Start
Prerequisites
Python 3.8 or higher
CUDA-compatible GPU (recommended for BERT model)
Installation
Clone the repository

git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
Create and activate a virtual environment

# Using conda
conda create -n BERT python=3.8
conda activate BERT

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

pip install -r requirements.txt
Download NLTK data

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
Running the Application
Start the Flask server

python app.py
Access the web interface

Open your browser and go to http://localhost:5000
Enter a news article in the text area
Click "Analyze Article" to get predictions
📁 Project Structure
fake-news-detection/
├── app.py                 # Main Flask application
├── models/                # Trained model files
│   ├── model.safetensors  # DistilBERT model
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   └── ...
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── static/                # Static files (CSS, JS)
└── requirements.txt       # Project dependencies
🛠️ Technical Details
Models Used
DistilBERT

Transformer-based model for deep semantic understanding
Handles contextual analysis and nuanced language patterns
Uses contextual understanding of language
Generally has higher accuracy than traditional ML models
Traditional ML Models

XGBoost: Gradient boosting for structured data
Random Forest: Ensemble of decision trees
LightGBM: Light gradient boosting machine
Logistic Regression: Linear classification
How It Works
Text Preprocessing

Lowercase conversion
Special character removal
Stopword removal
Stemming
Model Prediction

Each model analyzes the preprocessed text
BERT model uses transformer architecture
Other models use TF-IDF features
Consensus Voting System The system uses a weighted voting mechanism to determine the final prediction:

Vote Weighting:

BERT model gets 2 votes (counted twice)
All other models get 1 vote each
Total votes = Number of models + 1 (because BERT counts twice)
Example Calculation:

Models: BERT, XGBoost, Random Forest, LightGBM, Logistic Regression
Total votes = 6 (5 models + 1 extra for BERT)

If BERT predicts FAKE:
- BERT: 2 votes for FAKE
- Other models: 1 vote each
- Need > 3 votes (50% of 6) for FAKE consensus
Decision Making:

If more than 50% of total votes are "FAKE" → Final prediction is FAKE
Otherwise → Final prediction is REAL
Example Scenarios:

Scenario 1:
- BERT: FAKE (2 votes)
- XGBoost: FAKE (1 vote)
- Random Forest: FAKE (1 vote)
- LightGBM: REAL (1 vote)
- Logistic Regression: REAL (1 vote)
Total: 4 FAKE votes out of 6 → Consensus: FAKE

Scenario 2:
- BERT: REAL (2 votes)
- XGBoost: FAKE (1 vote)
- Random Forest: FAKE (1 vote)
- LightGBM: REAL (1 vote)
- Logistic Regression: REAL (1 vote)
Total: 3 FAKE votes out of 6 → Consensus: REAL
Why BERT gets 2 votes:

More sophisticated model architecture
Better at understanding context and nuance
Generally higher accuracy in practice
Can better handle complex language patterns
This weighted voting system ensures:

BERT's prediction has more influence
System is robust against individual model errors
Final decision considers both traditional ML and transformer-based approaches
🎯 Usage Example
# Example article
article = """
[Your news article text here]
"""

# Get predictions
results, consensus = predict_single_article(article)
print(f"Consensus: {consensus}")
for model, prediction in results.items():
    print(f"{model}: {prediction['label']} ({prediction['confidence']}%)")
📊 Performance
Accuracy: Varies by model (BERT typically highest)
Speed:
BERT: ~1-2 seconds per article
Other models: < 1 second per article
Memory Usage: ~500MB-1GB (mainly due to BERT model)
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face for the DistilBERT model
Flask for the web framework
All other open-source libraries used in this project
