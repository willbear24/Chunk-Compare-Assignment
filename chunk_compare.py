from sentence_transformers import SentenceTransformer, util
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

# A sample document comprised of multiple topics (500+ words)

document = """
FastAPI uses Pydantic (a Python library you'll learn in-depth in the next lesson) to validate incoming data automatically. Instead of writing validation code, you describe what valid data looks like.

The moment you define an endpoint in FastAPI, it appears in interactive documentation. 
FastAPI generates two documentation interfaces automatically:
Swagger UI (at /docs) — An interactive page where you can see all your endpoints, their parameters, their expected request bodies, and their response formats. You can even make live API calls directly from the browser.
ReDoc (at /redoc) — A cleaner, read-only documentation page that's great for sharing with other developers or teams.
This documentation is always in sync with your code because it's generated from your code. Change an endpoint, and the docs update automatically.

FastAPI is built on top of Starlette (an async web framework) and supports Python's async/await syntax natively. This means your API can handle multiple requests concurrently.

FastAPI uses Python's type hint system as actual functional code, not just comments.

FastAPI doesn't include a database layer — it's intentionally modular. You bring your own database library, and SQLAlchemy is the standard choice. Here's how the pieces fit together.

FastAPI has a powerful dependency injection system. Instead of manually creating and closing database sessions in every endpoint, you define a dependency function that FastAPI calls automatically.

Never store plain-text passwords. Passwords are hashed using bcrypt before storage. Even if the database is compromised, the passwords can't be recovered.

Tokens are stateless. The server doesn't store sessions. The token itself contains the user's identity, and the server can verify it using a secret key. This scales beautifully — any server in a cluster can validate any token.

Tokens expire. A token issued today shouldn't work forever. Setting an expiration (typically 30 minutes to 24 hours) limits the damage if a token is stolen.

The token encodes the user's identity (typically their ID or email) plus an expiration timestamp. The server signs it with a secret key. Anyone with the secret key can verify the token is legitimate and hasn't been tampered with.

Streamlit is a Python library that turns Python scripts into interactive web applications. You write Python, and Streamlit converts it into a fully functional web app with widgets, charts, tables, and layouts - all rendered in the browser.

Here’s the one thing you absolutely must understand about Streamlit, because everything else builds on it:
Every time a user interacts with any widget, the entire Python script re-runs from top to bottom.
It’s not like Flask or FastAPI where specific functions handle specific requests. In Streamlit, the whole script runs again. Every time.
User types in a text box? Script re-runs.
User clicks a button? Script re-runs.
User moves a slider? Script re-runs.
This is radically different from how JavaScript works. In JavaScript, you set up event listeners that run specific functions when specific things happen. In Streamlit, there are no event listeners - the entire script IS the event handler.
This design makes Streamlit incredibly simple to write, but it has one critical consequence: All variables reset to their initial values on every re-run.

Streamlit was designed for data and AI applications. It has built-in support for displaying DataFrames and tables with sorting and filtering, rendering charts with one line of code, streaming text (critical for AI chat interfaces), managing file uploads, caching expensive computations (like model loading or API calls), and chat interface components (st.chat_message, st.chat_input).
"""


# --- Strategy 1: Fixed-size chunks (300 chars, 50 overlap) ---
def fixed_chunks(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap  # Advance by a fixed step so this always terminates
    return [c for c in chunks if c]  # Remove empty chunks


# --- Strategy 2: Paragraph Chunks ---
def paragraph_chunks(text):
    paragraphs = text.strip().split("\n\n")  # Split on double new lines
    return [p.strip() for p in paragraphs if p.strip()]


strategies = {
    "Fixed (300 chars)": fixed_chunks(document),
    "Paragraph": paragraph_chunks(document),
}

for name, chunks in strategies.items():
    sizes = [len(c) for c in chunks]
    print(f"{name}: {len(chunks)} chunks, avg {sum(sizes)//len(sizes)} chars")

# --- Search Comparisons ---
queries = [
    "What is the purpose of FastAPI?",
    "How does Streamlit connect to FastAPI?",
    "What is the best way to store a JWT safely?",
]

query_embeddings = model.encode(queries)

for name, chunks in strategies.items():
    chunks_embeddings = model.encode(chunks)

    for query, query_embedding in zip(queries, query_embeddings):
        print(f"\nQuery: {query}")
        scores = util.cos_sim(query_embedding, chunks_embeddings)[0]
        top_results = scores.topk(k=min(2, scores.numel()))

        print(f"--- {name} ---")
        for rank, (score, index) in enumerate(
            zip(top_results.values, top_results.indices), start=1
        ):
            chunk_index = int(index.item())
            print(f"#{rank} score: {float(score.item()):.4f}")
            print(f"Best chunk ({len(chunks[chunk_index])} chars):")
            print(f"  '{chunks[chunk_index][:120]}...\n")

# Overall, I would say that the paragraph chunking performed better than the fixed size chunks. The cosine scores were slightly higher and the content was slightly more relevant and usable, although the information was nearly identical that the two methods pulled from the document. In some circumstances, for example the question about how JWT's should be stored, the first answer given in the fixed chunk actually seemed more relevant than that given by the paragraph chunk. But the paragraph chunk also had that information, it was just a close second in terms of its cosine score. There were also a couple of occasions where the context given by the overlap of the fixed chunk was actually helpful for understanding context. So in general, I would say that paragraph chunking was slightly more useful and clear, but fixed chunking has benefits in added context from overlap and in one instance provided a more accurate and useful result than paragraph chunking.
