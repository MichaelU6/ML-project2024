import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def test_category(samples, expected_values, vectorizer, pca, neighbors_model):
    true_count = 0
    for i, sample in enumerate(samples):
        predicted_next_word = predict_next_word(sample, vectorizer, pca, neighbors_model)
        if predicted_next_word == expected_values[i]:
            true_count += 1
    return true_count

def predict_next_word(input_text, vectorizer, pca, neighbors_model):
    # Vectorize the input text
    input_vector = vectorizer.transform([input_text]).toarray()

    # Transform the input vector using PCA
    input_vector_pca = pca.transform(input_vector)

    # Find the nearest neighbor in the reduced space
    _, nearest_neighbor_index = neighbors_model.kneighbors(input_vector_pca)

    # Get the original text corresponding to the nearest neighbor
    predicted_text = corpus[nearest_neighbor_index[0][0]]
    # Find the position of the input text in the predicted text
    input_position = predicted_text.find(input_text)

    next_word_index = input_position + len(input_text.split())  # Index of the word following the input text

    if next_word_index < len(predicted_text.split()):
        next_word = predicted_text.split()[next_word_index]
    else:
        next_word = False

    return next_word

def test_settings(vectorizer, pca, neighbors_model, categories, expected_values):
    results = {}
    x = 1
    for category, values in zip(categories, expected_values):
        true_count = test_category(category, values, vectorizer, pca, neighbors_model)
        accuracy = true_count / len(category)
        results[f'{x} Accuracy'] = accuracy
        x+=1
    return results

all_of_sentence_samples = ["klasifikácia obrazu je bežná aplikácia počítačového", "učenie s odmenami sa používa na trénovanie agentov na postupné", "analýza veľkých dát zahŕňa spracovanie a analýzu rozsiahlych sád", "prediktívne modelovanie pomáha predpovedať budúce trendy na základe historických", "klasifikácia s použitím support vector machines môže byť efektívna aj v priestoroch vyššej", "rozdielne krokové učenie upravuje veľkosť kroku na základe výkonu", "hierarchická zhlukovacia analýza skupuje dáta do stromovej", "evolučné algoritmy môžu byť využité pri hľadaní optimálnych", "hierarchické modely môžu modelovať vzťahy medzi príznakmi na viacerých", "Hintonove kapsulové siete sú navrhnuté na lepšie zachytávanie štruktúr v", "Shell skripty sa často používajú pre automatizáciu a spracovanie údajov v unixových", "Groovy, ako dynamický jazyk pre JVM, je populárny pre automatizáciu a", "Rust sa vyznačuje svojou bezpečnosťou a výkonom, čo ho robí vhodným pre kritické", "Dart je jazyk navrhnutý pre vývoj mobilných aplikácií a webových", "Groovy, ako jazyk pre Apache Groovy framework, uľahčuje integráciu s existujúcimi Java"]
start_of_sentence_samples = ["klasifikácia obrazu je bežná", "učenie s odmenami sa používa na trénovanie", "analýza veľkých dát zahŕňa spracovanie a analýzu", "prediktívne modelovanie pomáha", "klasifikácia s použitím support", "rozdielne krokové učenie upravuje", "hierarchická zhlukovacia", "evolučné algoritmy môžu byť", "hierarchické modely môžu modelovať", "Hintonove kapsulové siete sú navrhnuté na", "Shell skripty sa často používajú pre", "Groovy, ako dynamický jazyk pre", "Rust sa vyznačuje svojou", "Dart je jazyk navrhnutý pre vývoj mobilných", "Groovy, ako jazyk pre Apache Groovy"]
end_sentence_samples = ["aplikácia počítačového", "agentov na postupné", "spracovanie a analýzu rozsiahlych sád", "trendy na základe historických", "machines môže byť efektívna aj v priestoroch vyššej", "veľkosť kroku na základe výkonu", "analýza skupuje dáta do stromovej", "hľadaní optimálnych", "príznakmi na viacerých", "zachytávanie štruktúr v", "údajov v unixových", "pre automatizáciu a", "vhodným pre kritické", "aplikácií a webových", "existujúcimi Java"]
all_of_sentence_value = ["videnia", "rozhodovanie", "údajov", "údajov", "dimenzie", "modelu", "štruktúry", "hyperparametrov", "úrovniach", "dátach", "systémoch", "skriptovanie", "úlohy", "frameworkov", "projektmi"]
start_of_sentence_value = ["aplikácia", "agentov", "rozsiahlych", "predpovedať", "vector", "veľkosť", "analýza", "využité", "vzťahy", "lepšie", "automatizáciu", "JVM", "bezpečnosťou", "aplikácií", "framework"]
end_of_sentence_value = ["videnia", "rozhodovanie", "údajov", "údajov", "dimenzie", "modelu", "štruktúry", "hyperparametrov", "úrovniach", "dátach", "systémoch", "skriptovanie", "úlohy", "frameworkov", "projektmi"]


corpus = read_file("corpus.txt")

vectorizer = CountVectorizer()
pca = PCA(n_components=2)
neighbors_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')

# Test 30 different settings
for i in range(30):
    max_features = (i + 1) * 100
    n_components_pca = i + 1

    vectorizer.set_params(max_features=max_features)  # Modify CountVectorizer parameters
    pca.set_params(n_components=n_components_pca) 
    neighbors_model.set_params(algorithm='kd_tree')  # Modify NearestNeighbors parameters

    # Fit models with new settings
    X = vectorizer.fit_transform(corpus).toarray()
    X_pca = pca.fit_transform(X)
    neighbors_model.fit(X_pca)

    # Test and print results
    settings_results = test_settings(vectorizer, pca, neighbors_model, [all_of_sentence_samples, start_of_sentence_samples, end_sentence_samples], [all_of_sentence_value, start_of_sentence_value, end_of_sentence_value])

    #print(settings_results)