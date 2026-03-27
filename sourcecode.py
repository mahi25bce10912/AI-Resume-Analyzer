import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# SAMPLE TRAINING DATA
# -----------------------------
data = {
    "resume": [
        "Python machine learning data analysis pandas numpy",
        "Java spring boot backend developer microservices",
        "HTML CSS JavaScript frontend react developer",
        "Accounting finance taxation excel tally",
        "Machine learning deep learning AI NLP projects",
        "React node js mongodb full stack developer"
    ],
    "job_role": [
        "Data Scientist",
        "Backend Developer",
        "Frontend Developer",
        "Accountant",
        "Data Scientist",
        "Full Stack Developer"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df["cleaned"] = df["resume"].apply(clean_text)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["job_role"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# SKILL DATABASE
# -----------------------------
skills_db = [
    "python", "machine learning", "data analysis",
    "java", "spring", "react", "html", "css",
    "javascript", "accounting", "excel", "mongodb"
]

# -----------------------------
# ROLE-BASED SKILLS
# -----------------------------
role_skills = {
    "Data Scientist": ["python", "machine learning", "data analysis", "pandas"],
    "Backend Developer": ["java", "spring", "microservices"],
    "Frontend Developer": ["html", "css", "javascript", "react"],
    "Full Stack Developer": ["react", "node", "mongodb"],
    "Accountant": ["accounting", "excel", "tally"]
}

# -----------------------------
# ANALYSIS FUNCTION
# -----------------------------
def analyze_resume(resume_text):
    cleaned = clean_text(resume_text)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]

    found_skills = [skill for skill in skills_db if skill in resume_text.lower()]

    total_skills = len(skills_db)
    matched_skills = len(found_skills)
    score = int((matched_skills / total_skills) * 100)

    if score < 30:
        category = "Weak Resume"
    elif score < 60:
        category = "Average Resume"
    else:
        category = "Strong Resume"

    suggestions = []
    required = role_skills.get(prediction, [])

    for skill in required:
        if skill not in resume_text.lower():
            suggestions.append(f"Consider adding {skill} for {prediction} role")

    if score < 50:
        suggestions.append("Improve resume by adding more projects and skills")

    if not suggestions:
        suggestions.append("Your resume looks good. Try adding advanced skills.")

    return prediction, found_skills, suggestions, score, category, matched_skills, total_skills

# -----------------------------
# CLI FUNCTION
# -----------------------------
def main():
    print("\n==============================")
    print("   AI RESUME ANALYZER (CLI)   ")
    print("==============================")

    while True:
        print("\nChoose an option:")
        print("1. Analyze Resume")
        print("2. Exit")

        choice = input("Enter your choice (1/2): ").strip()

        if choice == "1":
            print("\nPaste your resume below:")
            resume_input = input(">> ")

            role, skills, suggestions, score, category, matched, total = analyze_resume(resume_input)

            print("\n===== ANALYSIS RESULT =====")
            print("🔹 Predicted Job Role:", role)
            print("🔹 Reason: Based on detected skills in the resume")
            print("🔹 Resume Score:", score, "/100")
            print("🔹 Category:", category)
            print("🔹 Skills Found:", skills)

            print("\n🔹 Suggestions:")
            for s in suggestions:
                print(" -", s)

            # -----------------------------
            # GRAPH (Non-blocking)
            # -----------------------------
            missing = total - matched

            plt.figure()
            plt.bar(['Found Skills', 'Missing Skills'], [matched, missing], color=['yellow', 'pink'])
            plt.title("Skill Match Analysis of Resume")
            plt.xlabel("Skill Type")
            plt.ylabel("Count")

            plt.savefig("resume_analysis.png")

            # Non-blocking graph
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        elif choice == "2":
            print("\nExiting... Thank you for using AI Resume Analyzer!")
            break

        else:
            print("\nInvalid choice. Please enter 1 or 2.")

# -----------------------------
# RUN PROGRAM
# -----------------------------
if __name__ == "__main__":
    main()
