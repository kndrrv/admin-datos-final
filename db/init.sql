CREATE TABLE IF NOT EXISTS raw_data (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak FLOAT,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    target INTEGER,
    loaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS curated_data (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak FLOAT,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    target INTEGER,
    processed_at TIMESTAMP DEFAULT NOW()
);