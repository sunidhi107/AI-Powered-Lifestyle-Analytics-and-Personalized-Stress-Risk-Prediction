from __future__ import annotations

import json
import sqlite3
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    session,
    url_for
)

from werkzeug.security import (
    check_password_hash,
    generate_password_hash
)

import database


# ============================================================
# APP CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_model"

app = Flask(__name__)

app.secret_key = "stressrisk-ai-development-key-change-in-production"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL = None
FEATURES = []

try:
    model_path = MODEL_DIR / "stressrisk_model.pkl"
    feature_path = MODEL_DIR / "feature_columns.pkl"

    # Load the complete Random Forest pipeline saved by the notebook.
    # The pipeline must contain preprocessing + classifier.
    MODEL = joblib.load(model_path)

    # Load the exact raw feature order used during training.
    FEATURES = list(joblib.load(feature_path))

    if len(FEATURES) != 19:
        raise ValueError(
            f"Expected exactly 19 model features, but found {len(FEATURES)}: {FEATURES}"
        )

    required_removed = {"Meal Regularity", "Meditation Practice"}
    if required_removed.intersection(FEATURES):
        raise ValueError(
            "Old features found in feature_columns.pkl: "
            f"{sorted(required_removed.intersection(FEATURES))}"
        )

    print("======================================")
    print("StressRisk AI Model Loaded Successfully")
    print("Model file:", model_path)
    print("Number of Features:", len(FEATURES))
    print("Features:", FEATURES)
    print("======================================")

except Exception as e:
    MODEL = None
    FEATURES = []
    print("======================================")
    print("MODEL LOADING ERROR")
    print(e)
    print("======================================")


# ============================================================
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(view):

    @wraps(view)
    def wrapped(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped


# ============================================================
# DATABASE SETUP
# ============================================================

def ensure_database():

    database.create_tables()

    connection = database.get_connection()

    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(assessments)"
        )
    }

    additions = {

        "stress_level": "INTEGER",

        "created_at": "TEXT",

        "health_data": "TEXT",

        "recommendations": "TEXT",

        "top_features": "TEXT"
    }

    for name, kind in additions.items():

        if name not in columns:

            connection.execute(
                f"ALTER TABLE assessments ADD COLUMN {name} {kind}"
            )

    connection.commit()

    connection.close()


# ============================================================
# CURRENT USER
# ============================================================

def current_user():

    if "user_id" not in session:
        return None

    connection = database.get_connection()

    user = connection.execute(
        "SELECT * FROM users WHERE id = ?",
        (session["user_id"],)
    ).fetchone()

    connection.close()

    return user


# ============================================================
# LATEST ASSESSMENT
# ============================================================

def latest_assessment():

    connection = database.get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM assessments
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (session["user_id"],)
    ).fetchone()

    connection.close()

    return dict(row) if row else {}


# ============================================================
# SAFE FLOAT CONVERSION
# ============================================================

def safe_float(value, label):

    try:

        number = float(value)

        if not np.isfinite(number):
            raise ValueError

        return number

    except (TypeError, ValueError):

        raise ValueError(
            f"Please enter a valid value for {label}."
        )


# ============================================================
# CONVERT FORM DATA TO MODEL FEATURES
# ============================================================

def form_to_features(form):
    required = {
        "gender": "Gender",
        "age": "Age",
        "occupation": "Occupation",
        "sleep_duration": "Sleep Duration",
        "quality_of_sleep": "Quality of Sleep",
        "physical_activity": "Physical Activity Level",
        "heart_rate": "Heart Rate",
        "daily_steps": "Daily Steps",
        "systolic_bp": "Systolic BP",
        "diastolic_bp": "Diastolic BP",
        "bmi": "BMI",
        "bmi_category": "BMI Category",
        "working_hours": "Working Hours",
        "screen_time": "Screen Time",
        "water_intake": "Water Intake",
        "caffeine_intake": "Caffeine Intake",
        "social_interaction": "Social Interaction",
        "smoking_habit": "Smoking Habit",
        "alcohol_intake": "Alcohol Intake"
    }

    missing = [
        label
        for name, label in required.items()
        if not str(form.get(name, "")).strip()
    ]

    if missing:
        if len(missing) == 1:
            raise ValueError(f"Please complete the missing field: {missing[0]}.")
        raise ValueError(
            f"Please complete all required fields ({len(missing)} missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})."
        )

    # Normalize string inputs
    gender = str(form.get("gender", "")).strip()
    occupation = str(form.get("occupation", "")).strip().title()
    bmi_category = str(form.get("bmi_category", "")).strip()
    smoking_habit = str(form.get("smoking_habit", "")).strip()
    alcohol_intake = str(form.get("alcohol_intake", "")).strip()

    values = {
        "Gender": gender,
        "Age": safe_float(form.get("age"), "Age"),
        "Occupation": occupation,
        "Sleep Duration": safe_float(form.get("sleep_duration"), "Sleep Duration"),
        "Quality of Sleep": safe_float(form.get("quality_of_sleep"), "Quality of Sleep"),
        "Physical Activity Level": safe_float(form.get("physical_activity"), "Physical Activity Level"),
        "Heart Rate": safe_float(form.get("heart_rate"), "Heart Rate"),
        "Daily Steps": safe_float(form.get("daily_steps"), "Daily Steps"),
        "Systolic BP": safe_float(form.get("systolic_bp"), "Systolic Blood Pressure"),
        "Diastolic BP": safe_float(form.get("diastolic_bp"), "Diastolic Blood Pressure"),
        "BMI": safe_float(form.get("bmi"), "BMI"),
        "BMI Category": bmi_category,
        "Working Hours": safe_float(form.get("working_hours"), "Working Hours"),
        "Screen Time": safe_float(form.get("screen_time"), "Screen Time"),
        "Water Intake": safe_float(form.get("water_intake"), "Water Intake"),
        "Caffeine Intake": safe_float(form.get("caffeine_intake"), "Caffeine Intake"),
        "Social Interaction": safe_float(form.get("social_interaction"), "Social Interaction"),
        "Smoking Habit": smoking_habit,
        "Alcohol Intake": alcohol_intake
    }

    # Make the dataframe using the EXACT feature order saved during training.
    if not FEATURES:
        raise ValueError("The trained model feature list is not available.")

    missing_model_values = [
        feature for feature in FEATURES
        if feature not in values
    ]

    if missing_model_values:
        raise ValueError(
            "The assessment form is missing model features: "
            + ", ".join(missing_model_values)
        )

    extra_removed_features = {
        "Meal Regularity",
        "Meditation Practice"
    }.intersection(FEATURES)

    if extra_removed_features:
        raise ValueError(
            "The saved model still expects removed features: "
            + ", ".join(sorted(extra_removed_features))
        )

    return pd.DataFrame(
        [{feature: values[feature] for feature in FEATURES}],
        columns=FEATURES
    )


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_for(score):

    score = float(score)

    if score >= 7:

        return "High Risk"

    if score >= 4:

        return "Medium Risk"

    return "Low Risk"


# ============================================================
# EXPLAIN PREDICTION (SHAP + FEATURE IMPORTANCE FALLBACK)
# ============================================================

EXPLAINER = None

def get_tree_explainer():
    global EXPLAINER
    if EXPLAINER is None and MODEL is not None:
        try:
            import shap
            estimator = MODEL.named_steps.get("model") or MODEL.named_steps.get("classifier")
            if estimator is not None:
                EXPLAINER = shap.TreeExplainer(estimator)
        except Exception as exc:
            print("Notice: TreeExplainer initialization fallback:", exc)
            EXPLAINER = False
    return EXPLAINER if EXPLAINER is not False else None


def explain_prediction(features, prediction):
    """
    Computes Explainable AI feature contributions using SHAP TreeExplainer on
    the actual trained model, with graceful fallback to RandomForest feature importances.
    """
    if MODEL is None:
        return []

    try:
        preprocessor = MODEL.named_steps.get("preprocessor")
        estimator = MODEL.named_steps.get("model") or MODEL.named_steps.get("classifier")

        if preprocessor is None or estimator is None:
            return []

        transformed = preprocessor.transform(features)
        feature_names = preprocessor.get_feature_names_out()

        explainer = get_tree_explainer()
        if explainer is not None:
            sv = explainer.shap_values(transformed)

            classes = list(estimator.classes_)
            class_idx = classes.index(prediction) if prediction in classes else 0

            if isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_vals = sv[0, :, class_idx]
            elif isinstance(sv, list):
                shap_vals = sv[class_idx][0]
            else:
                shap_vals = sv[0]

            feat_dict = {}
            for name, val in zip(feature_names, shap_vals):
                clean = name.replace("numeric__", "").replace("categorical__", "")
                # Map one-hot encoded category columns back to original feature name
                for raw_col in FEATURES:
                    if clean == raw_col or clean.startswith(raw_col + "_"):
                        feat_dict[raw_col] = feat_dict.get(raw_col, 0.0) + float(val)
                        break
                else:
                    feat_dict[clean] = feat_dict.get(clean, 0.0) + float(val)

            sorted_feats = sorted(feat_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            max_val = max(abs(v) for _, v in sorted_feats) if sorted_feats else 1.0
            max_val = max(max_val, 1e-9)

            return [
                {
                    "Feature": name,
                    "SHAP": round(val, 4),
                    "Absolute": round(abs(val) / max_val, 4)
                }
                for name, val in sorted_feats
            ]

    except Exception as shap_exc:
        print("SHAP computation notice:", shap_exc)

    # Fallback to model feature importances
    try:
        estimator = MODEL.named_steps.get("model") or MODEL.named_steps.get("classifier")
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            names = MODEL.named_steps["preprocessor"].get_feature_names_out().tolist()
            transformed = MODEL.named_steps["preprocessor"].transform(features)
            transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
            values = transformed_array[0] * np.asarray(importances)

            feat_dict = {}
            for name, val in zip(names, values):
                clean = name.replace("numeric__", "").replace("categorical__", "")
                for raw_col in FEATURES:
                    if clean == raw_col or clean.startswith(raw_col + "_"):
                        feat_dict[raw_col] = feat_dict.get(raw_col, 0.0) + float(val)
                        break
                else:
                    feat_dict[clean] = feat_dict.get(clean, 0.0) + float(val)

            sorted_feats = sorted(feat_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            max_val = max(abs(v) for _, v in sorted_feats) if sorted_feats else 1.0
            max_val = max(max_val, 1e-9)

            return [
                {
                    "Feature": name,
                    "SHAP": round(val, 4),
                    "Absolute": round(abs(val) / max_val, 4)
                }
                for name, val in sorted_feats
            ]
    except Exception as fallback_exc:
        print("Feature importance fallback notice:", fallback_exc)

    return []


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(risk, features):
    """
    Generates personalized, non-diagnostic lifestyle and wellness suggestions
    based on the user's specific assessment metrics and predicted risk tier.
    """
    recs = []

    if risk == "High Risk":
        recs.append(
            "Consider speaking with a qualified healthcare professional about your current stress and wellbeing."
        )

    row = features.iloc[0]
    sleep_duration = float(row.get("Sleep Duration", 8))
    quality_of_sleep = float(row.get("Quality of Sleep", 7))
    physical_activity = float(row.get("Physical Activity Level", 45))
    daily_steps = float(row.get("Daily Steps", 8000))
    working_hours = float(row.get("Working Hours", 8))
    screen_time = float(row.get("Screen Time", 5))
    water_intake = float(row.get("Water Intake", 2.5))
    caffeine_intake = float(row.get("Caffeine Intake", 2))
    smoking_habit = str(row.get("Smoking Habit", "No"))
    alcohol_intake = str(row.get("Alcohol Intake", "None"))

    if sleep_duration < 7.0:
        recs.append(
            "Establish a consistent sleep schedule and a 30–45 minute screen-free wind-down routine to support restorative sleep."
        )
    elif quality_of_sleep <= 5:
        recs.append(
            "Optimize your rest environment by keeping your bedroom dark, quiet, and cool to improve subjective sleep quality."
        )

    if physical_activity < 30.0:
        recs.append(
            "A short daily walk of 20–30 minutes or gentle movement stimulates endorphins and releases accumulated physical tension."
        )
    elif daily_steps < 5000:
        recs.append(
            "Incorporate light movement breaks throughout the day to gradually increase your daily step count."
        )

    if screen_time > 6.0:
        recs.append(
            "Practice the 20-20-20 rule during prolonged screen use and set a digital curfew at least 30 minutes before bedtime."
        )

    if working_hours > 8.5:
        recs.append(
            "Set clear cognitive boundaries between work hours and personal time, taking short 2-minute restorative pauses."
        )

    if caffeine_intake > 3.0:
        recs.append(
            "Moderate daily caffeine consumption and observe an early afternoon cutoff (e.g. 2 PM) to preserve circadian sleep depth."
        )

    if water_intake < 2.0:
        recs.append(
            "Aim for 2.0 to 2.5 litres of water across the day, as mild dehydration directly amplifies physiological stress responses."
        )

    if smoking_habit == "Yes":
        recs.append(
            "Consider reducing nicotine consumption and trying deep diaphragmatic breathing when feeling acute stress urges."
        )

    if alcohol_intake in ["Frequent", "Occasional"] and risk in ["High Risk", "Medium Risk"]:
        recs.append(
            "Limiting alcohol intake helps stabilize heart rate variability and prevents fragmented deep sleep."
        )

    recs.append(
        "Keep a consistent sleep and nutrition routine, and review changes in your stress score over time."
    )

    return recs


# ============================================================
# PAGE CONTEXT
# ============================================================

def page_context(
    user,
    **extra
):

    latest = latest_assessment()

    health = {}


    if latest.get("health_data"):

        try:

            health = json.loads(
                latest["health_data"]
            )

        except (
            TypeError,
            json.JSONDecodeError
        ):

            health = latest


    prediction = (

        latest.get("stress_level")

        or latest.get("stress_prediction")

        or 0
    )


    context = {

        "user_name":
            user["name"]
            if user
            else "User",

        "user_email":
            user["email"]
            if user
            else "",

        "user_role":
            "User",

        "prediction":
            int(float(prediction))
            if prediction
            else 0,

        "risk_level":
            latest.get(
                "risk_level",
                "Low Risk"
            ),

        "assessment_count":
            0,

        "sleep_duration":
            health.get(
                "sleep_duration",
                "-"
            ),

        "quality_of_sleep":
            health.get(
                "quality_of_sleep",
                "-"
            ),

        "heart_rate":
            health.get(
                "heart_rate",
                "-"
            ),

        "daily_steps":
            health.get(
                "daily_steps",
                "-"
            ),

        "health":
            health
    }


    connection = database.get_connection()

    context["assessment_count"] = connection.execute(

        """
        SELECT COUNT(*)
        FROM assessments
        WHERE user_id = ?
        """,

        (session.get("user_id"),)

    ).fetchone()[0]

    connection.close()


    context.update(extra)

    return context


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "index.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    error = None


    if request.method == "POST":

        connection = database.get_connection()

        user = connection.execute(

            """
            SELECT *
            FROM users
            WHERE email = ?
            """,

            (
                request.form
                .get(
                    "email",
                    ""
                )
                .strip()
                .lower(),
            )

        ).fetchone()

        connection.close()


        password_ok = (

            user

            and (

                check_password_hash(

                    user["password"],

                    request.form.get(
                        "password",
                        ""
                    )
                )

                or

                user["password"]
                == request.form.get(
                    "password",
                    ""
                )
            )
        )


        if password_ok:

            session.update(

                user_id=user["id"],

                name=user["name"],

                role="User"
            )

            return redirect(
                url_for("dashboard")
            )


        error = (
            "Incorrect email or password."
        )


    return render_template(
        "login.html",
        error=error
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    error = None


    if request.method == "POST":

        if (

            request.form.get("password")
            !=
            request.form.get(
                "confirm_password"
            )
        ):

            error = (
                "Passwords do not match."
            )

        else:

            try:

                connection = (
                    database.get_connection()
                )


                connection.execute(

                    """
                    INSERT INTO users
                    (name, email, password, gender, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,

                    (

                        request.form
                        .get(
                            "name",
                            ""
                        )
                        .strip(),

                        request.form
                        .get(
                            "email",
                            ""
                        )
                        .strip()
                        .lower(),

                        generate_password_hash(

                            request.form.get(
                                "password",
                                ""
                            )
                        ),

                        request.form.get(
                            "gender"
                        ),

                        datetime.now()
                        .isoformat(
                            timespec="seconds"
                        )
                    )
                )


                connection.commit()

                connection.close()


                return redirect(
                    url_for("login")
                )


            except sqlite3.IntegrityError:

                error = (
                    "An account with that email already exists."
                )


    return render_template(
        "register.html",
        error=error
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(

        "dashboard.html",

        **page_context(
            current_user()
        )
    )


# ============================================================
# ASSESSMENT PAGE
# ============================================================

@app.route("/assessment")
@login_required
def assessment():

    return render_template(

        "assessment.html",

        prefill={},

        prefill_source="",

        **page_context(
            current_user()
        )
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    if MODEL is None:

        return render_template(

            "assessment.html",

            prefill=request.form,

            error=(
                "The trained model could not be loaded. "
                "Please check the saved_model folder."
            ),

            **page_context(
                current_user()
            )
        )


    try:

        # Convert form input into model dataframe

        features = form_to_features(
            request.form
        )


        # Debug information before prediction
        print("\n========== PREDICTION DEBUG ==========")
        print("Raw request.form:", request.form.to_dict(flat=True))
        print("Model feature count:", len(FEATURES))
        print("Model features:", FEATURES)
        print(
            "Model expected input:",
            getattr(MODEL, "feature_names_in_", "unavailable")
        )
        print("Input columns:", features.columns.tolist())
        print("Input dtypes:")
        print(features.dtypes)
        print("Input values:")
        print(features.to_dict(orient="records")[0])
        print(
            "Categorical values:",
            {
                name: features[name].tolist()
                for name in features.select_dtypes(exclude="number").columns
            }
        )
        print(
            "Numerical values:",
            {
                name: features[name].tolist()
                for name in features.select_dtypes(include="number").columns
            }
        )
        print("Missing values:", features.isna().sum().to_dict())
        print("======================================\n")

        # Make prediction
        prediction = int(
            MODEL.predict(features)[0]
        )


        # Convert score into risk category

        risk = risk_for(
            prediction
        )


        # Store health information

        health = {

            "age":
                request.form.get("age"),

            "gender":
                request.form.get("gender"),

            "occupation":
                request.form.get("occupation"),

            "sleep_duration":
                request.form.get(
                    "sleep_duration"
                ),

            "quality_of_sleep":
                request.form.get(
                    "quality_of_sleep"
                ),

            "physical_activity":
                request.form.get(
                    "physical_activity"
                ),

            "bmi_category":
                request.form.get(
                    "bmi_category"
                ),

            "bmi":
                request.form.get("bmi"),

            "working_hours":
                request.form.get(
                    "working_hours"
                ),

            "screen_time":
                request.form.get(
                    "screen_time"
                ),

            "water_intake":
                request.form.get(
                    "water_intake"
                ),

            "caffeine_intake":
                request.form.get(
                    "caffeine_intake"
                ),

           

            "social_interaction":
                request.form.get(
                    "social_interaction"
                ),

           

            "smoking_habit":
                request.form.get(
                    "smoking_habit"
                ),

            "alcohol_intake":
                request.form.get(
                    "alcohol_intake"
                ),

            "heart_rate":
                request.form.get(
                    "heart_rate"
                ),

            "daily_steps":
                request.form.get(
                    "daily_steps"
                ),

            "systolic_bp":
                request.form.get(
                    "systolic_bp"
                ),

            "diastolic_bp":
                request.form.get(
                    "diastolic_bp"
                ),

            "stress_level":
                prediction
        }


        # Generate recommendations

        recs = build_recommendations(

            risk,

            features
        )


        # Generate feature explanations

        top_features = explain_prediction(

            features,

            prediction
        )


        # Save assessment

        connection = (
            database.get_connection()
        )


        connection.execute(

            """
            INSERT INTO assessments
            (
                user_id,
                age,
                gender,
                occupation,
                sleep_duration,
                quality_of_sleep,
                physical_activity,
                bmi_category,
                heart_rate,
                daily_steps,
                systolic_bp,
                diastolic_bp,
                stress_prediction,
                risk_level,
                assessment_date,
                stress_level,
                created_at,
                health_data,
                recommendations,
                top_features
            )
            VALUES
            (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,

            (

                session["user_id"],

                health["age"],

                health["gender"],

                health["occupation"],

                health["sleep_duration"],

                health["quality_of_sleep"],

                health["physical_activity"],

                health["bmi_category"],

                health["heart_rate"],

                health["daily_steps"],

                health["systolic_bp"],

                health["diastolic_bp"],

                prediction,

                risk,

                datetime.now().isoformat(
                    timespec="seconds"
                ),

                prediction,

                datetime.now().isoformat(
                    timespec="seconds"
                ),

                json.dumps(health),

                json.dumps(recs),

                json.dumps(top_features)
            )
        )


        connection.commit()

        connection.close()


        # Show result page

        return render_template(

            "result.html",

            **page_context(

                current_user(),

                prediction=prediction,

                risk_level=risk,

                health=health,

                recommendations=recs,

                top_features=top_features
            )
        )


    except ValueError as exc:

        return render_template(

            "assessment.html",

            prefill=request.form,

            error=str(exc),

            **page_context(
                current_user()
            )
        )


    except Exception as exc:

        print("\n========== PREDICTION ERROR ==========")
        print("ERROR:", repr(exc))
        traceback.print_exc()
        print("======================================\n")

        # Development-friendly error message so the actual problem is visible
        # instead of hiding it behind a generic message.
        return render_template(

            "assessment.html",

            prefill=request.form,

            error=f"Prediction Error: {exc}",

            **page_context(
                current_user()
            )
        )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    connection = (
        database.get_connection()
    )


    rows = connection.execute(

        """
        SELECT *
        FROM assessments
        WHERE user_id = ?
        ORDER BY id DESC
        """,

        (session["user_id"],)

    ).fetchall()


    connection.close()


    return render_template(

        "history.html",

        history=[
            dict(row)
            for row in rows
        ],

        **page_context(
            current_user()
        )
    )


# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics")
@login_required
def analytics():

    connection = (
        database.get_connection()
    )


    rows = connection.execute(

        """
        SELECT *
        FROM assessments
        WHERE user_id = ?
        ORDER BY id ASC
        """,

        (session["user_id"],)

    ).fetchall()


    connection.close()


    return render_template(

        "analytics.html",

        history=[
            dict(row)
            for row in rows
        ],

        **page_context(
            current_user()
        )
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.route("/recommendations")
@login_required
def recommendations():

    latest = latest_assessment()


    if latest.get("recommendations"):

        recs = json.loads(
            latest["recommendations"]
        )

    else:

        recs = [

            "Complete an assessment to receive a wellness recommendation."
        ]


    return render_template(

        "recommendations.html",

        recommendations=recs,

        icons=["✓"] * len(recs),

        **page_context(
            current_user()
        )
    )


# ============================================================
# MINDFULNESS
# ============================================================

@app.route("/mindfulness")
@login_required
def mindfulness():

    tips = [

        (
            "Box breathing",
            "A calm reset",
            "Try four counts in, hold, out, and hold."
        ),

        (
            "Gentle movement",
            "Reconnect with your body",
            "A short comfortable walk can create space in a busy day."
        )
    ]


    return render_template(

        "mindfulness.html",

        tip=tips[0],

        tips=tips,

        **page_context(
            current_user()
        )
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
@login_required
def profile():

    user = current_user()


    return render_template(

        "profile.html",

        gender=user["gender"],

        **page_context(user)
    )


# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings")
@login_required
def settings():

    return render_template(

        "settings.html",

        **page_context(
            current_user()
        )
    )


# ============================================================
# CLEAR HEALTH DATA
# ============================================================

@app.route(
    "/api/health-data/clear",
    methods=["POST"]
)
@login_required
def api_health_data_clear():

    return redirect(
        url_for("assessment")
    )





# ============================================================
# INITIALIZE DATABASE
# ============================================================

ensure_database()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )