from preprocess_predict import PredictionPipeline


def _make_pipeline():
    return PredictionPipeline(
        "models/best_model.pkl",
        "models/preprocessor.pkl",
    )


def test_preprocess_shape():
    pipeline = _make_pipeline()
    data = {
        "temperature": 120.0,
        "pressure": 2.5,
        "flow_rate": 150.0,
        "vibration": 0.8,
    }
    X_scaled = pipeline.preprocess(data)
    assert X_scaled.shape == (1, len(pipeline.features_order))


def test_predict_proba_between_0_and_1():
    pipeline = _make_pipeline()
    data = {
        "temperature": 120.0,
        "pressure": 2.5,
        "flow_rate": 150.0,
        "vibration": 0.8,
    }
    proba = pipeline.predict_proba(data)
    assert 0.0 <= proba <= 1.0
