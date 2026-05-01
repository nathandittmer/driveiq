from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from driveiq.evaluation.dataset import IntentExample, load_intent_examples


@dataclass
class IntentPrediction:
    query: str
    predicted_intent: str
    confidence: float


def train_intent_classifier(
    examples: list[IntentExample],
) -> Pipeline:
    queries = [example.query for example in examples]
    labels = [example.intent for example in examples]

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(queries, labels)
    return model


def predict_intent(model: Pipeline, query: str) -> IntentPrediction:
    predicted_label = model.predict([query])[0]

    confidence = 0.0
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba([query])[0]
        confidence = float(max(probabilities))

    return IntentPrediction(
        query=query,
        predicted_intent=str(predicted_label),
        confidence=confidence,
    )


def load_and_train_default_intent_classifier(
    dataset_path: str = "data/eval/intent_eval.json",
) -> Pipeline:
    examples = load_intent_examples(dataset_path)
    return train_intent_classifier(examples)