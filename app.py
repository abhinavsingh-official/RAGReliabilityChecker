"""RAGReliabilityChecker - most RAG demos build a chatbot and stop. This
checks whether a RAG answer is actually grounded in what was retrieved,
flagging the exact unsupported spans instead of trusting the answer."""

import html

import streamlit as st

from hallucination_checker import check_answer, load_detector
from retriever import Retriever

st.set_page_config(page_title="RAGReliabilityChecker", layout="wide")

PAGE_STYLE = """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }
h1 { font-weight: 700; letter-spacing: -0.02em; color: #15803d; }
.rr-panel {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.rr-chunk {
    background: #f8fafc; border-left: 3px solid #94a3b8; padding: 0.5rem 0.8rem;
    margin-bottom: 0.4rem; font-size: 0.92rem; color: #334155;
}
.rr-flag { background: #fecaca; border-bottom: 2px solid #b91c1c; }
</style>
"""

DEMO_PAIRS = {
    "France - grounded": (
        "What is the capital of France and its population?",
        "The capital of France is Paris, and the population is about 68 million.",
    ),
    "France - hallucinated number": (
        "What is the capital of France and its population?",
        "The capital of France is Paris, and the population is about 70 million, "
        "making it the most populous country in Europe.",
    ),
    "Everest - grounded": (
        "How tall is Mount Everest and who first summited it?",
        "Mount Everest stands at 8,849 meters and was first summited in 1953 by "
        "Edmund Hillary and Tenzing Norgay.",
    ),
    "Everest - hallucinated claim": (
        "How tall is Mount Everest and who first summited it?",
        "Mount Everest stands at 8,849 meters and was first summited in 1921 by "
        "George Mallory, who reached the summit solo.",
    ),
}


@st.cache_resource
def _load():
    return Retriever(), load_detector()


def _highlight(answer: str, spans) -> str:
    if not spans:
        return html.escape(answer)
    spans = sorted(spans, key=lambda s: s.start)
    out, cursor = [], 0
    for s in spans:
        out.append(html.escape(answer[cursor:s.start]))
        out.append(f'<span class="rr-flag">{html.escape(answer[s.start:s.end])}</span>')
        cursor = s.end
    out.append(html.escape(answer[cursor:]))
    return "".join(out)


def home_tab():
    st.markdown(
        '<div class="rr-panel" style="padding:2rem 2.2rem;">'
        "<h2 style='margin-top:0;'>Checks whether a RAG answer is actually grounded</h2>"
        "<p style='color:#374151; font-size:1.05rem; max-width:760px; line-height:1.6;'>"
        "Most RAG portfolio projects build a chatbot and stop - the answer looks plausible, "
        "so it ships. This does the harness work instead: retrieve the source chunks "
        "(TF-IDF over a small document set), then run a span-level hallucination detector "
        "over the answer against that exact context, flagging unsupported claims instead "
        "of trusting the model's confidence."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#4b5563;">Head to <b>Check</b> to try a built-in grounded/'
        "hallucinated pair, or paste your own question, context, and answer.</p>",
        unsafe_allow_html=True,
    )


def check_tab():
    retriever, detector = _load()

    mode = st.radio("Input", ["Built-in example", "Your own"], horizontal=True)

    if mode == "Built-in example":
        choice = st.selectbox("Example", list(DEMO_PAIRS.keys()))
        question, answer = DEMO_PAIRS[choice]
        st.text_input("Question", value=question, disabled=True)
        st.text_area("Answer to check", value=answer, height=80, disabled=True)
    else:
        question = st.text_input("Question", value="What is the capital of France and its population?")
        answer = st.text_area("Answer to check", height=80)
        if not answer:
            return

    context = retriever.retrieve(question, k=3)

    st.markdown("#### Retrieved context")
    for chunk in context:
        st.markdown(f'<div class="rr-chunk">{html.escape(chunk)}</div>', unsafe_allow_html=True)

    with st.spinner("Checking groundedness..."):
        spans = check_answer(detector, context, question, answer)

    st.markdown("#### Answer, with unsupported spans flagged")
    st.markdown(
        f'<div class="rr-panel">{_highlight(answer, spans)}</div>', unsafe_allow_html=True
    )

    if spans:
        st.warning(f"{len(spans)} unsupported span(s) found - not grounded in the retrieved context.")
        for s in spans:
            st.caption(f"\"{s.text.strip()}\" - confidence {s.confidence:.0%}")
    else:
        st.success("No unsupported spans found - the answer is grounded in the retrieved context.")


def main():
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.title("RAGReliabilityChecker")
    st.caption("Span-level hallucination detection for RAG answers, traced to source chunks.")
    tab0, tab1 = st.tabs(["Home", "Check"])
    with tab0:
        home_tab()
    with tab1:
        check_tab()


if __name__ == "__main__":
    main()
