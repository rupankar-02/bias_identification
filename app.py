import streamlit as st
import joblib
import numpy as np
# Create Streamlit app
import torch
import transformers as ppb

def encode_input_data(input_data):
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    # Tokenize the new data
    tokenized_data = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in input_data]
    max_len = max([len(seq) for seq in tokenized_data])
    padded_data = np.array([seq + [0] * (max_len - len(seq)) for seq in tokenized_data])
    attention_mask = np.where(padded_data != 0, 1, 0)
    input_ids = torch.tensor(padded_data)
    attention_mask = torch.tensor(attention_mask)

    print("Encoding the input data")
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    return features


st.title("Bias Detection")
option = st.selectbox(
    'Choose a model to run:',
    ('-------------------------SELECT A MODEL--------------------------------','Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'MLP')
)
if option != '-------------------------SELECT A MODEL--------------------------------':
    st.write('You selected:', option)

user_input = st.text_area('Enter your thoughts here')

# biases={'country_bias': 'Country Bias', 'religion_bias': 'Religion Bias',gender_bias:'Gender Bias',2:"Non Bias"}


if st.button("Predict"):
    if user_input:
        try:
            model = joblib.load(f"model/{option}.pkl")
            # Reshape the input to be a 2D array

            input_array = np.array([user_input])

            input_array = encode_input_data(input_array)

            predicted_sentiment = model.predict(input_array)
  
            predicted_sentiment_label = predicted_sentiment[0]
            st.info(f"Predicted Bias: {predicted_sentiment_label}")
        except FileNotFoundError:
            st.error(f"The model file '{option}.pkl' was not found.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter some text to analyze.")