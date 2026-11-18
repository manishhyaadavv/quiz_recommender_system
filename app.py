import random
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from config import get_db_connection
from dotenv import load_dotenv
import os
import requests
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")


@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# REGISTRATION
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form["full_name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if not full_name or not email or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must be at least 8 characters!", "danger")
            return redirect(url_for("register"))

        connection = get_db_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                "SELECT id FROM users WHERE email = %s",
                (email,)
            )

            existing_user = cursor.fetchone()

            if existing_user:
                flash("Email already registered!", "danger")
                return redirect(url_for("register"))

            hashed_password = generate_password_hash(password)

            cursor.execute(
                """INSERT INTO users
                (full_name, email, password_hash)
                VALUES (%s, %s, %s)""",
                (full_name, email, hashed_password)
            )

            connection.commit()

            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))

        finally:
            cursor.close()
            connection.close()

    return render_template("register.html")


# LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip().lower()
        password = request.form["password"]

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        try:
            cursor.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )

            user = cursor.fetchone()

            if user and check_password_hash(
                user["password_hash"], password
            ):

                session["user_id"] = user["id"]
                session["user_name"] = user["full_name"]
                session["user_role"] = user["role"]

                return redirect(url_for("dashboard"))

            flash("Invalid email or password!", "danger")

        finally:
            cursor.close()
            connection.close()

    return render_template("login.html")


# DASHBOARD
@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("dashboard.html")


# LOGOUT
@app.route("/logout")
def logout():

    session.clear()
    flash("You have logged out successfully.", "success")

    return redirect(url_for("login"))

@app.route("/quizzes")
def quizzes():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("SELECT * FROM topics")
    topics = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template("quizzes.html", topics=topics)

@app.route("/quiz/<int:topic_id>", methods=["GET", "POST"])
def take_quiz(topic_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT * FROM topics WHERE id = %s",
            (topic_id,)
        )
        topic = cursor.fetchone()

        if not topic:
            return "Topic not found", 404

        questions = []

        # Python ke liye QuizAPI use hogi
        if topic_id in [1, 2, 5]:

            if request.method == "GET":

                try:
                    api_key = os.getenv("QUIZ_API_KEY")

                    response = requests.get(
                        "https://quizapi.io/api/v1/questions",
                        headers={
                            "Authorization": f"Bearer {api_key}"
                        },
                        params={
                            "category": "Programming",
                            "tags": {
                            1: "python",
                            2: "java",
                            5: "javascript"
                            }[topic_id],
                            "type": "MULTIPLE_CHOICE",
                            "limit": 50,
                            "random": "true"
                        },
                        timeout=20
                    )

                    response.raise_for_status()

                    api_questions = response.json().get("data", [])

                    for item in api_questions:

                        answers = item.get("answers", [])

                        correct_answer = next(
                            (
                                i for i, answer in enumerate(answers)
                                if answer.get("isCorrect") is True
                            ),
                            None
                        )

                        if correct_answer is None or len(answers) < 2:
                            continue

                        question = {
                            "id": str(item["id"]),
                            "question_text": item["text"],
                            "option_a": answers[0]["text"],
                            "option_b": answers[1]["text"],
                            "option_c": answers[2]["text"] if len(answers) > 2 else "",
                            "option_d": answers[3]["text"] if len(answers) > 3 else "",
                            "correct_option": chr(65 + correct_answer)
                        }

                        questions.append(question)

                    if questions:
                        random.shuffle(questions)

                        questions = questions[:10]

                        session["active_api_quiz"] = questions
                        session["active_api_topic"] = topic_id

                except (requests.RequestException, ValueError, KeyError):
                    questions = []

            else:
                # POST par wahi questions use honge jo GET par aaye the
                if session.get("active_api_topic") == topic_id:
                    questions = session.get("active_api_quiz", [])

        # API unavailable hone par MySQL se questions
        if not questions:

            cursor.execute("""
                SELECT id, question_text, option_a, option_b,
                option_c, option_d, correct_option
                FROM questions
                WHERE topic_id = %s
            """, (topic_id,))

            questions = cursor.fetchall()

        score = None

        if request.method == "POST":

            score = 0

            for question in questions:
                selected = request.form.get(str(question["id"]))

                if selected == question["correct_option"]:
                    score += 1

            cursor.execute("""
                INSERT INTO quiz_attempts
                (user_id, topic_id, score, total_questions)
                VALUES (%s, %s, %s, %s)
            """, (
                session["user_id"],
                topic_id,
                score,
                len(questions)
            ))

            connection.commit()

        # Correct answers frontend par nahi bhejne
        display_questions = [
            {
                key: value
                for key, value in question.items()
                if key != "correct_option"
            }
            for question in questions
        ]

        return render_template(
            "take_quiz.html",
            topic=topic,
            questions=display_questions,
            score=score
        )

    finally:
        cursor.close()
        connection.close()

# PERSONALIZED RECOMMENDATIONS
@app.route("/recommendations")
def recommendations():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            topics.id,
            topics.topic_name,
            COUNT(quiz_attempts.id) AS total_attempts,
            ROUND(
                AVG(
                    100.0 * quiz_attempts.score /
                    NULLIF(quiz_attempts.total_questions, 0)
                ), 1
            ) AS average_score
        FROM topics
        LEFT JOIN quiz_attempts
            ON topics.id = quiz_attempts.topic_id
            AND quiz_attempts.user_id = %s
        GROUP BY topics.id, topics.topic_name
        ORDER BY
            CASE
                WHEN COUNT(quiz_attempts.id) = 0 THEN 0
                ELSE 1
            END,
            average_score ASC
    """, (session["user_id"],))

    topic_performance = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        "recommendations.html",
        topic_performance=topic_performance
    )

# MY PERFORMANCE
@app.route("/performance")
def performance():
    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            COUNT(*) AS total_attempts,
            ROUND(AVG(100.0 * score / NULLIF(total_questions, 0)), 1) AS average_score,
            ROUND(MAX(100.0 * score / NULLIF(total_questions, 0)), 1) AS highest_score
        FROM quiz_attempts
        WHERE user_id = %s
    """, (session["user_id"],))

    stats = cursor.fetchone()

    cursor.execute("""
        SELECT
            topics.topic_name,
            quiz_attempts.score,
            quiz_attempts.total_questions,
            quiz_attempts.attempted_at
        FROM quiz_attempts
        JOIN topics ON quiz_attempts.topic_id = topics.id
        WHERE quiz_attempts.user_id = %s
        ORDER BY quiz_attempts.attempted_at DESC
    """, (session["user_id"],))

    attempts = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        "performance.html",
        stats=stats,
        attempts=attempts
    )

if __name__ == "__main__":
    app.run(debug=True)