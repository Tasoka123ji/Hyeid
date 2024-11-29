from flask import Flask, render_template, request, jsonify, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from test import read_text_from_file

client = Groq(api_key='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')
vectorizer = TfidfVectorizer()
app = Flask(__name__)





def foo(query, top_k=2):

    document_text = read_text_from_file('data.txt')
    
    tfidf_matrix = vectorizer.fit_transform([query] + document_text)
    query_vector = tfidf_matrix[0]  
    text_vectors = tfidf_matrix[1:]  
    similarities = cosine_similarity(query_vector, text_vectors).flatten()
    
    top_k_indices = similarities.argsort()[-top_k:][::-1]  
    top_matches = [(document_text[i], similarities[i]) for i in top_k_indices]
    
    result = f"Question: {query}\n\nTop {top_k} Matches:\n"
    for idx, (doc, score) in enumerate(top_matches):
        result += f"\n{idx + 1}. Score: {score:.2f}\n{doc}\n"
    print(result)
    print('------------------------------------------------------------------------')

    return result


def promt(text):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": f"{text}"
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    k = ''
    for chunk in completion:
        if type((chunk.choices[0].delta.content)) is str:
            k += (chunk.choices[0].delta.content)
        

    return k

@app.route("/")
def hello():
    return render_template('chat.html')



@app.route("/get")
def get_bot_response():
    print('get')
    message = request.args.get('msg')
    text = foo(message)
    print(message)
    result = promt(text)
    return result
        

def process_message(message):
    # Example processing logic
    print('foo')
    return f"You said: {message}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)




