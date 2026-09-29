"""
Syntax + route sanity check for the StressRisk AI project.
Run from the project root:  python _check_syntax.py
"""
import ast
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
ok = True


def check_python(rel_path):
    global ok
    p = ROOT / rel_path
    try:
        ast.parse(p.read_text(encoding="utf-8"))
        print(f"  PASS  {rel_path}")
    except SyntaxError as e:
        print(f"  FAIL  {rel_path}  line {e.lineno}: {e.msg}")
        ok = False


def grep(rel_path, pattern, label, must_exist=True):
    global ok
    p = ROOT / rel_path
    found = bool(re.search(pattern, p.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL))
    status = "PASS" if (found == must_exist) else "FAIL"
    if status == "FAIL":
        ok = False
    verdict = "found" if found else "missing"
    print(f"  {status}  {label}  ({verdict} in {rel_path})")


# ── 1. Python syntax ─────────────────────────────────────────
print("\n=== Python syntax ===")
check_python("database.py")
check_python("app.py")
check_python("train_model.py")

# ── 2. All active routes in app.py ──────────────────────────
print("\n=== Routes present in app.py ===")
routes = [
    ("/", "home route"),
    ("/login", "login route"),
    ("/register", "register route"),
    ("/logout", "logout route"),
    ("/dashboard", "dashboard route"),
    ("/assessment", "assessment route"),
    ("/predict", "predict route"),
    ("/history", "history route"),
    ("/analytics", "analytics route"),
    ("/recommendations", "recommendations route"),
    ("/mindfulness", "mindfulness route"),
    ("/profile", "profile route"),
    ("/settings", "settings route"),
    ("/api/health-data/clear", "health data clear route"),
]

app_content = (ROOT / "app.py").read_text(encoding="utf-8")
for route_path, label in routes:
    pattern = r'@app\.route\s*\(\s*["\']' + re.escape(route_path) + r'["\']'
    grep("app.py", pattern, label)

# ── 3. Assessment template checks ───────────────────────────
print("\n=== assessment.html checks ===")
grep("templates/assessment.html", r'name="gender"',             "gender field present")
grep("templates/assessment.html", r'name="age"',                "age field present")
grep("templates/assessment.html", r'name="occupation"',         "occupation field present")
grep("templates/assessment.html", r'name="sleep_duration"',     "sleep_duration field present")
grep("templates/assessment.html", r'name="quality_of_sleep"',   "quality_of_sleep field present")
grep("templates/assessment.html", r'name="physical_activity"',  "physical_activity field present")
grep("templates/assessment.html", r'name="daily_steps"',        "daily_steps field present")
grep("templates/assessment.html", r'name="heart_rate"',         "heart_rate field present")
grep("templates/assessment.html", r'name="systolic_bp"',        "systolic_bp field present")
grep("templates/assessment.html", r'name="diastolic_bp"',       "diastolic_bp field present")
grep("templates/assessment.html", r'name="bmi"',                "bmi field present")
grep("templates/assessment.html", r'name="bmi_category"',       "bmi_category field present")
grep("templates/assessment.html", r'name="working_hours"',      "working_hours field present")
grep("templates/assessment.html", r'name="screen_time"',        "screen_time field present")
grep("templates/assessment.html", r'name="water_intake"',       "water_intake field present")
grep("templates/assessment.html", r'name="caffeine_intake"',    "caffeine_intake field present")
grep("templates/assessment.html", r'name="social_interaction"', "social_interaction field present")
grep("templates/assessment.html", r'name="smoking_habit"',      "smoking_habit field present")
grep("templates/assessment.html", r'name="alcohol_intake"',     "alcohol_intake field present")
grep("templates/assessment.html", r'action="{{ url_for\([\'"]predict[\'"]\) }}"', "form action -> /predict")

# ── Summary ──────────────────────────────────────────────────
print()
print("=" * 42)
print("RESULT:", "ALL CHECKS PASSED [PASS]" if ok else "ONE OR MORE CHECKS FAILED [FAIL]")
print("=" * 42)
sys.exit(0 if ok else 1)
