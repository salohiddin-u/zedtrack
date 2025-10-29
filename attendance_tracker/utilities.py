from django.http import JsonResponse
from attendance_tracker.models import Attendance

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import vertexai
from vertexai.generative_models import GenerativeModel


def predicted_attendance(request):
    data = Attendance.objects.filter(center=request.user).values('student_id', 'time', 'status')
    df = pd.DataFrame.from_records(data)

    if df.empty:
        return JsonResponse({"error": "⚠️ No attendance data found"})

    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Convert attendance status to binary (1 = Present, 0 = Absent)
    if df['status'].dtype == object:
        df['present'] = df['status'].map({"Present": 1, "Absent": 0})
    else:
        df['present'] = df['status'].astype(int)

    # Calculate weekly average attendance rate
    weekly = df.groupby(df["time"].dt.isocalendar().week)['present'].mean() * 100

    # Train a simple linear regression model
    X = np.arange(len(weekly)).reshape(-1, 1)
    y = weekly.values
    model = LinearRegression().fit(X, y)

    # Predict next week's attendance
    model_step_next = len(weekly)
    iso_week_next = weekly.index.max() + 1
    pred = model.predict([[model_step_next]])[0]
    last_week_rate = weekly.iloc[-1]

    return [float(pred), float(last_week_rate)]


vertexai.init(project="zedtrack-ai-assistant", location="us-central1")
insight_model = GenerativeModel("gemini-2.5-flash-lite")


def insights(request):
    records = Attendance.objects.filter(center=request.user).values("student", "time", "status", "id")

    data_text = "\n".join([f"{r['student']} ({r['id']}) - {r['time']} - {r['status']}" for r in records])

    predicted, last_week = predicted_attendance(request)
    predicted_rounded = round(predicted, 2)
    last_week_rounded = round(last_week, 2)

    prompt = f"""
    You are an assistant that analyzes attendance data.
    ONLY use the data below, do not invent or assume values.

    Here is attendance data:
    {data_text}

    Tasks:
    1. Highlight students with attendance below 70% in last 4 records.
    2. Give essential highlights.

    Rules:
    - Use ONLY the attendance data provided.
    - Do not add extra comments or make up information.
    - If data is missing, say "Not enough data".
    - Don't change the percentages ({predicted_rounded} and {last_week_rounded}), which are provided in the text.
    - Follow exactly this format:

    <b>Attendance report:</b><br>
    <span style="color: var(--main-color);"><b>-</b></span> Attendance rate for last week:
    <span style="color: var(--main-color);"><b>{last_week_rounded}%</b></span><br>
    <span style="color: var(--main-color);"><b>-</b></span> Predicted attendance rate for next week:
    <span style="color: var(--main-color);"><b>{predicted_rounded}%</b></span><br>
    <b>Essential highlights:</b><br>
    <span style="color: var(--main-color);"><b>-</b></span> first highlight.<br>
    <span style="color: var(--main-color);"><b>-</b></span> second highlight.<br>
    <span style="color: var(--main-color);"><b>-</b></span> third highlight.<br>
    """

    response = insight_model.generate_content(prompt)
    return response.text
