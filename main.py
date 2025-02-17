from flask import Flask, request, jsonify
import pdfplumber
import psycopg2
import os
# import spacy
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

app = Flask(__name__)

# PostgreSQL Connection Details
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "rutgers1234",
    "host": "resume-screener-db.cd2ysagkm5vw.ap-southeast-2.rds.amazonaws.com",
    "port": "5432",
}

openai.api_key = "xxxx"

# Load NLP Models
# nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index for Resume Embeddings
index = faiss.IndexFlatL2(384)  # 384 is the embedding dimension
resumes = []  # Store resume texts
resume_embeddings = []  # Store resume embeddings


# Function to connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text


# Function to extract skills (currently detecting ORG entities, but can be fine-tuned)
def extract_skills(resume_text):
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "ORG"]  # Adjust label for better accuracy
    return skills


# Function to embed text
def embed_text(text):
    return embed_model.encode(text)


# Load resumes from the database into FAISS index
def load_resumes_to_faiss():
    global resumes, resume_embeddings, index
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, text FROM public.resumes;")
        fetched_resumes = cur.fetchall()
        cur.close()
        conn.close()

        resumes = [r[1] for r in fetched_resumes]
        resume_embeddings = np.array([embed_text(res) for res in resumes])

        if not index.is_trained:
            index = faiss.IndexFlatL2(384)  # Reinitialize if needed
        index.add(resume_embeddings)

        print(f"Loaded {len(resumes)} resumes into FAISS.")

    except Exception as e:
        print(f"Error loading resumes into FAISS: {e}")


@app.route("/upload_resume/", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)

    file.save(file_path)

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(file_path)

    # Insert into PostgreSQL
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO public.resumes (filename, text) VALUES (%s, %s) RETURNING id;",
                    (file.filename, extracted_text))
        resume_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        # Update FAISS index with the new resume
        resumes.append(extracted_text)
        new_embedding = embed_text(extracted_text).reshape(1, -1)
        index.add(new_embedding)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"resume_id": resume_id, "filename": file.filename, "text": extracted_text[:500]})  # Returning first 500 characters


@app.route("/extract_skills/", methods=["GET"])
def get_resume_skills():
    """Fetch resumes from DB, extract skills, and return results."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch all resumes
        cur.execute("SELECT id, text FROM public.resumes;")
        resumes = cur.fetchall()

        cur.close()
        conn.close()

        # Extract skills for each resume
        results = []
        for resume_id, resume_text in resumes:
            skills = extract_skills(resume_text)
            results.append({
                "resume_id": resume_id,
                "skills": skills
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/match_resume/", methods=["POST"])
def match_resume():
    """Find the best matching resume for a job description."""
    job_description = request.json.get("job_description")
    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    try:
        job_embedding = embed_text(job_description).reshape(1, -1)
        D, I = index.search(job_embedding, k=1)

        best_match_resume = resumes[I[0][0]]
        return jsonify({"best_match_resume": best_match_resume})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/refine_match/", methods=["POST"])
def refine_match():
    """Use GPT-4 to refine the match between a resume and job description."""
    job_text = request.json.get("job_text")
    resume_text = request.json.get("resume_text")

    if not job_text or not resume_text:
        return jsonify({"error": "Both job_text and resume_text are required"}), 400

    try:

        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Match the resume with the job description"},
                {"role": "user", "content": f"Resume: {resume_text}, Job: {job_text}"}
            ],
            max_tokens=500,
            temperature=0.5
        )
        response = chat_response.choices[0].message.content.strip()


        return jsonify({"refined_match": chat_response.choices[0].message.content.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_resumes_to_faiss()  # Load existing resumes on startup
    app.run(debug=True, port=6666)
