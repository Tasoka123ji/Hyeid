from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

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
    # print(result)
    
    return result



def read_text_from_file(file_path):
    print('---------------------------------------------------------------------------------')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            arr = text.split('......')
        return arr
    except FileNotFoundError:
        return "Error: The file was not found."
    except Exception as e:
        return f"Error: {e}"

    
    

# print(foo('hi how i can changing password '))

# print(len(read_text_from_file('data.txt')))
