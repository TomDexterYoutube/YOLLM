from difflib import SequenceMatcher
from config import SIMILARITY_SENSITIVITY
def check_similarity(generated_text, training_data):
    similarity = SequenceMatcher(None, generated_text, training_data).ratio()
    
    # If similarity is too high (>80%), likely direct copying
    if similarity > SIMILARITY_SENSITIVITY:
        return False, "Output too similar to training data"
    return True, "Output appears to be original"

def validate_generation(text, training_samples):
    for sample in training_samples:
        is_valid, message = check_similarity(text, sample)
        if not is_valid:
            return False, message
    return True, "Generation passed validation"
