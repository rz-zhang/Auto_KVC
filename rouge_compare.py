from evaluate import load

def calculate_rouge_hf(predictions, references):
    rouge = load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

# Test predictions and references
predictions_hf = ["The article discusses the challenges faced by ex-offenders in finding accommodation in Wales. A charity, Prison Link Cymru, reported that some ex-offowners were living rough for up to a year before finding suitable accommodation. The charity and others argued that investment in housing would be cheaper than jailing homeless repeat offenders. The article highlights the difficulties in finding accommodation, particularly for men, and the need for more affordable housing. The Welsh Government has implemented measures to prevent homelessness among prisoners, but more needs to be done to address the issue. A local charity, Emmaus, provides accommodation and support to homeless individuals, and highlights the importance of connecting people with the services they need. The article concludes that building more one-bedroom flats could help ease the problem. Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark and share this content Bookmark"]
references_hf = ["There is a \"chronic\" need for more housing for prison leavers in Wales, according to a charity.",]

# Calculate and print scores
rouge_scores_hf = calculate_rouge_hf(predictions_hf, references_hf)
print("Hugging Face Evaluate ROUGE Scores:", rouge_scores_hf)


from rouge import Rouge

def calculate_rouge_pypi(predictions, references):
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references)
    return scores

# Test predictions and references
predictions_pypi = ["the quick brown fox jumps over the lazy dog"]
references_pypi = ["the quick brown fox jumped over the lazy dog"]

# Calculate and print scores
rouge_scores_pypi = calculate_rouge_pypi(predictions_hf, references_hf)
print("PyPI ROUGE Package Scores:", rouge_scores_pypi)
