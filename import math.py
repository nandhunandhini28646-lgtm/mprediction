import math

# ---------- PREDICTIVE MODEL ----------
def predict_risk(work_hours, sleep, exercise_val, wlb,
                 sad, fatigue, alcohol_val, smoking_val, social_support):

    # normalize continuous vars (same as JS)
    work_score = (work_hours - 40) * 0.06
    sleep_score = (7.0 - sleep) * 0.35
    exercise_score = (1.0 - exercise_val) * 0.5
    wlb_score = (5 - wlb) * 0.25
    sad_score = (sad - 5) * 0.2
    fatigue_score = (fatigue - 5) * 0.18
    alcohol_score = alcohol_val * 0.25
    smoking_score = smoking_val * 0.3
    social_score = (5 - social_support) * 0.2

    logit = (-0.8 + work_score + sleep_score + exercise_score +
             wlb_score + sad_score + fatigue_score +
             alcohol_score + smoking_score + social_score)

    prob = 1 / (1 + math.exp(-logit))

    # clip between 0.12 and 0.92
    prob = max(0.12, min(prob, 0.92))

    return prob


# ---------- Example Input ----------
probability = predict_risk(
    work_hours=45,
    sleep=5.5,
    exercise_val=0.5,
    wlb=4,
    sad=7,
    fatigue=6,
    alcohol_val=0.6,
    smoking_val=0,
    social_support=5
)

percent = round(probability * 100)

# Risk level
if percent < 35:
    risk = "LOW"
elif percent < 60:
    risk = "MODERATE"
else:
    risk = "ELEVATED"

print("Mental Health Risk:", risk)
print("Probability:", percent, "%")
print("Odds:", round(probability/(1-probability), 2), ":1")
