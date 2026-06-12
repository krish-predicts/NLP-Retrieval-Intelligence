NLP Retrieval Intelligence Platform
Production-grade NLP Retrieval Intelligence Platform built on the Amazon Fine Food Reviews dataset. Transforms unstructured reviews into actionable insights for Product, Marketing, Customer Experience, Operations, and Leadership teams.

Features
Data pipeline: ingestion, validation, deduplication, stratified sampling, preprocessing, EDA
Retrieval: TF-IDF, BM25, dense (SentenceTransformers + FAISS), hybrid
Insights: BERTopic, sentiment alignment, KeyBERT, feature request mining, defect detection
Evaluation: Precision@K, Recall@K, MRR, nDCG, business coverage metrics
Dashboard: 8-page Streamlit app with search, analytics, and benchmarks
Experiments: Jupyter notebook comparing retrieval strategies
Project Structure
project/
├── data/raw|processed|embeddings/
├── notebooks/retrieval_experiments.ipynb
├── src/ingestion|preprocessing|retrieval|embeddings|insights|evaluation|dashboard/
├── tests/
├── reports/
├── app.py
└── requirements.txt
Setup
cd project
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
Place Reviews.csv in data/raw/ (Amazon Fine Food Reviews schema).

Configuration
All paths and hyperparameters are config-driven via environment variables or .env:

Variable	Default	Description
SAMPLE_SIZE	50000	Stratified sample size; 0 = full corpus
RANDOM_SEED	42	Reproducibility
DEFAULT_EMBEDDING_MODEL	all-mpnet-base-v2	Dense retrieval model
HYBRID_ALPHA	0.5	BM25 weight in hybrid search
PROJECT_ROOT	auto-detected	Override project root path
Run Pipeline
cd project
python -m src.pipelines.run_all                  # full pipeline
python -m src.pipelines.run_all --stage ingestion
python -m src.pipelines.run_all --stage evaluation
Outputs (reports/)
retrieved_results.csv — retrieval results for eval queries
topic_summary.csv — BERTopic summary
feature_requests.csv — mined feature requests
defect_report.csv — defect frequency and alerts
stakeholder_summary.pdf — executive stakeholder report
retrieval_benchmark.csv — model comparison metrics
Launch Dashboard
cd project
streamlit run app.py
From the UI:

Sentiment Analysis page → click Run Sentiment Analysis
Retrieval Benchmark page → click Run Retrieval Benchmark
Or run stages from the terminal:

python -m src.pipelines.run_all --stage sentiment
python -m src.pipelines.run_all --stage evaluation
Run Tests
cd project
pytest tests/ -v
Experiments
Open notebooks/retrieval_experiments.ipynb for:

TF-IDF vs BM25
BM25 vs Dense
Dense vs Hybrid
Chunking strategies (fixed, overlap, semantic)
Embedding comparison (mpnet vs bge)
Evaluation Notes
Retrieval metrics use pseudo-qrels (phrase match + BM25 ranking) because no human-labeled relevance judgments exist. Lexical retrievers may score higher on these metrics; hybrid search typically performs best on semantic queries.

Business Questions Answered
Why are ratings increasing/decreasing?
What are the most common complaints?
What features do customers request?
What drives positive reviews?
What product defects are emerging?
Which topics correlate with low ratings?
Tech Stack
Python, Pandas, NumPy, Scikit-Learn, SentenceTransformers, FAISS, Rank-BM25, BERTopic, KeyBERT, Plotly, Streamlit, Pydantic, Pytest
