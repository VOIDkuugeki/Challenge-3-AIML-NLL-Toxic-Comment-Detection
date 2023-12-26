from flask import Flask, render_template, request
from transformers import AutoTokenizer, TFBertForSequenceClassification
import tensorflow as tf

app = Flask(__name__, template_folder='templates')

phobert_tokenizer = AutoTokenizer.from_pretrained("path-to-save/Tokenizer")
phobert_model = TFBertForSequenceClassification.from_pretrained("path-to-save/Model")

label = {
    0: 'An toàn',
    1: 'Mang tính công kích',
    2: 'Độc hại'
}

def get_sentiment(review, tokenizer=phobert_tokenizer, model=phobert_model):
    if not isinstance(review, list):
        review = [review]

    input_ids, token_type_ids, attention_mask = tokenizer.batch_encode_plus(
        review,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='tf'
    ).values()

    prediction = model.predict([input_ids, token_type_ids, attention_mask-1])
    pred_labels = tf.argmax(prediction.logits, axis=1)
    pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
    return pred_labels

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {'sentiment_label': '', 'message': ''}
    
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment_label = get_sentiment(user_input)

        result = {
            'sentiment_label': sentiment_label[0],
            'message': '',
        }

        if sentiment_label[0] == 'An toàn':
            result['message'] = 'Câu này an toàn.'
        elif sentiment_label[0] == 'Mang tính công kích':
            result['message'] = 'Câu này mang tính công kích.'
        elif sentiment_label[0] == 'Độc hại':
            result['message'] = 'Câu này độc hại.'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
