import json
import difflib
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load external resources
with open("allergen_dictionary.json") as f:
    allergen_dict = json.load(f)

with open("reverse_synonym_index.json") as f:
    reverse_synonyms = json.load(f)

# Flatten allergen terms for fuzzy matching
allergen_terms = list(reverse_synonyms.keys())

# Load BioBERT NER pipeline
tokenizer = AutoTokenizer.from_pretrained("final_biobert_model")
model = AutoModelForTokenClassification.from_pretrained("final_biobert_model")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def normalize_ci(text):
    return re.sub(r"(ci|cl|c\.i\.)\s?(\d+)", r"CI\2", text, flags=re.IGNORECASE)

def clean_text(text):
    text = normalize_ci(text)
    return re.sub(r"[^\w\s\-(),.&]", "", text)

def get_closest_match(term, threshold=0.8):
    matches = difflib.get_close_matches(term, allergen_terms, n=1, cutoff=threshold)
    return matches[0] if matches else None

def resolve_canonical(term):
    term_lower = term.lower()
    for k, v in allergen_dict.items():
        if k.lower() == term_lower or term_lower in [s.lower() for s in v.get("synonyms", [])]:
            return k
    return reverse_synonyms.get(term)

def detect_allergens(text):
    cleaned = clean_text(text)
    entities = ner_pipeline(cleaned)
    raw_terms = list(set(ent["word"] for ent in entities))

    normalized_map = {}
    matched_allergens = []
    fuzzy_matches = []

    for term in raw_terms:
        norm_term = re.sub(r"[^a-zA-Z0-9\- ]", "", term.lower())
        canonical = resolve_canonical(norm_term)
        if canonical:
            matched_allergens.append(canonical)
            normalized_map[term] = canonical
        else:
            fuzzy = get_closest_match(norm_term)
            if fuzzy:
                canonical = reverse_synonyms.get(fuzzy)
                if canonical:
                    fuzzy_matches.append((term, fuzzy))
                    matched_allergens.append(canonical)
                    normalized_map[term] = canonical
                else:
                    normalized_map[term] = None
            else:
                normalized_map[term] = None

    # Deduplicate
    matched_allergens = list(set(matched_allergens))

    # Compute overall risk
    risk = "NONE"
    if matched_allergens:
        risks = {allergen_dict[a]["risk"] for a in matched_allergens if a in allergen_dict}
        if "High" in risks:
            risk = "HIGH"
        elif "Moderate" in risks:
            risk = "MODERATE"
        elif "Low" in risks:
            risk = "LOW"

    return {
        "risk_level": risk,
        "matched_allergens": matched_allergens,
        "normalized_map": normalized_map,
        "detected_chemicals": [k for k, v in normalized_map.items() if v],
        "fuzzy_matches": fuzzy_matches
    }
    
# Export allergen dictionary for external access
def get_allergen_dict():
    return allergen_dict