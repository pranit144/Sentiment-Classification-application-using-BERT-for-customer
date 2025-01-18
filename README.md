# Sentiment Classifier with BERT

This project is a web application that uses a fine-tuned BERT model for sentiment classification. The application is built using Streamlit, and it allows users to input text and receive predictions on whether the sentiment is positive or negative.

## Features

- **Sentiment Analysis**: Classifies input text as positive or negative using a BERT model.
- **Interactive UI**: Built with Streamlit for a user-friendly experience.
- **Text Statistics**: Displays word and character count for the input text.
- **Prediction Scores**: Shows positive and negative sentiment scores.
- **Visualization**: Bar chart visualization of sentiment scores.

## Demo

### Screenshot
![Sentiment Classifier Demo](https://via.placeholder.com/800x400?text=Screenshot+of+the+application)

## Installation

Follow these steps to set up and run the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-classifier.git
   cd sentiment-classifier
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or higher installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Pretrained Model and Tokenizer**:
   - Place your fine-tuned BERT model in `C:/Users/Pranit/PycharmProjects/customer/Model`.
   - Place your tokenizer files in `C:/Users/Pranit/PycharmProjects/customer/Tokenizer`.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Model/                 # Directory for the fine-tuned BERT model
├── Tokenizer/             # Directory for the tokenizer files
└── README.md              # Project documentation
```

## Dependencies

- `streamlit`
- `transformers`
- `tensorflow`
- `numpy`

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Enter a sentence in the text area.
2. Click the **Classify Sentiment** button.
3. View the predicted sentiment, scores, and analysis.

## Customization

- **Model Path**: Update the model and tokenizer paths in the `load_model` function if your files are stored elsewhere.
- **CSS Styling**: Modify the `st.markdown` section in `app.py` to change the UI appearance.

## Example

### Input
```
I love this product! It's amazing and works perfectly.
```

### Output
- Predicted Sentiment: **Positive**
- Positive Score: 0.95
- Negative Score: 0.05
- Word Count: 8
- Character Count: 47

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or open an issue.

## Contact

For questions or support, please contact [your_email@example.com](mailto:your_email@example.com).

---

⭐ **If you find this project helpful, please give it a star!** ⭐
