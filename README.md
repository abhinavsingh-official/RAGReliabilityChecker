# RAGReliabilityChecker

Most RAG portfolio projects build a chatbot and stop. This checks whether
a RAG answer is actually grounded in what was retrieved, flagging the
exact unsupported spans instead of trusting the model's confidence.

## Screenshots

![Home](screenshots/home.png)
![Check](screenshots/check.png)

## How it works

1. A small factual document set is chunked and retrieved with TF-IDF
   given a question (`retriever.py`) - the "R" in this RAG pipeline.
2. A candidate answer (built-in example, or your own) is checked against
   that retrieved context with a span-level hallucination detector
   ([LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect), MIT),
   which flags exactly which substrings of the answer aren't supported.
3. Flagged spans are highlighted directly in the answer, with a confidence
   score each, so you can see precisely what's unsupported and why.

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

The hallucination detector downloads a pretrained model (~500MB) from
Hugging Face on first run.

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT, see `LICENSE`.
