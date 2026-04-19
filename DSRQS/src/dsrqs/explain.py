def explain_prediction(score, threshold):
    if score > threshold:
        return "Relevant relation (high confidence)"
    elif score > threshold * 0.7:
        return "Moderate relevance"
    else:
        return "Irrelevant relation"