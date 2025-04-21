import streamlit as st
import pandas as pd
import requests
import json 
if __name__ == "__main__":
    import sys
    sys.path.append('code')
    from apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition
else:
    from code.apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition

PLACE_IDS_SOURCE_FILE = "cache/place_ids.csv"
CACHE_REVIEWS_FILE = "cache/reviews.csv"
CACHE_SENTIMENT_FILE = "cache/reviews_sentiment_by_sentence.csv"
CACHE_ENTITIES_FILE = "cache/reviews_sentiment_by_sentence_with_entities.csv"


def reviews_step(place_ids: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(place_ids, str):
        place_ids = pd.read_csv(place_ids)

    all_reviews = []

    for _, row in place_ids.iterrows():
        place_id = row["place_id"]
        place_info = get_google_place_details(place_id)

        if "result" in place_info and "reviews" in place_info["result"]:
            reviews = place_info["result"]["reviews"]
            for review in reviews:
                review_entry = {
                    "place_id": place_id,
                    "name": place_info["result"].get("name", ""),
                    "author_name": review.get("author_name", ""),
                    "rating": review.get("rating", None),
                    "text": review.get("text", "")
                }
                all_reviews.append(review_entry)

    df = pd.DataFrame(all_reviews)
    df.to_csv(CACHE_REVIEWS_FILE, index=False)
    return df


def sentiment_step(reviews: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(reviews, str):
        reviews = pd.read_csv(reviews)

    all_sentences = []

    for _, row in reviews.iterrows():
        sentiment_data = get_azure_sentiment(row["text"])
        doc = sentiment_data["results"]["documents"][0]

        for sentence in doc["sentences"]:
            sentence_entry = {
                "place_id": row["place_id"],
                "name": row["name"],
                "author_name": row["author_name"],
                "rating": row["rating"],
                "sentence_text": sentence["text"],
                "sentence_sentiment": sentence["sentiment"],
                "confidenceScores.positive": sentence["confidenceScores"]["positive"],
                "confidenceScores.neutral": sentence["confidenceScores"]["neutral"],
                "confidenceScores.negative": sentence["confidenceScores"]["negative"]
            }
            all_sentences.append(sentence_entry)

    df = pd.DataFrame(all_sentences)
    df.to_csv(CACHE_SENTIMENT_FILE, index=False)
    return df


def entity_extraction_step(sentences: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(sentences, str):
        sentences = pd.read_csv(sentences)

    all_entities = []

    for _, row in sentences.iterrows():
        ner_data = get_azure_named_entity_recognition(row["sentence_text"])
        doc = ner_data["results"]["documents"][0]

        for entity in doc.get("entities", []):
            entity_entry = {
                "place_id": row["place_id"],
                "name": row["name"],
                "author_name": row["author_name"],
                "rating": row["rating"],
                "sentence_text": row["sentence_text"],
                "sentence_sentiment": row["sentence_sentiment"],
                "confidenceScores.positive": row["confidenceScores.positive"],
                "confidenceScores.neutral": row["confidenceScores.neutral"],
                "confidenceScores.negative": row["confidenceScores.negative"],
                "entity_text": entity.get("text", ""),
                "entity_category": entity.get("category", ""),
                "entity_subCategory": entity.get("subCategory", ""),
                "confidenceScores.entity": entity.get("confidenceScore", 0)
            }
            all_entities.append(entity_entry)

    df = pd.DataFrame(all_entities)
    df.to_csv(CACHE_ENTITIES_FILE, index=False)
    return df

if __name__ == '__main__':
    # helpful for debugging as you can view your dataframes and json outputs
    import streamlit as st 
    st.write("What do you want to debug?")