# 📊 ZEDTRACK — Effortless Attendance. Powerful Insights

<image src="static/image.png">
ZedTrack is a web-based attendance management system built with **Django** that helps educational centers track student attendance, analyze behavior, and visualize key metrics. It is built for your university portfolio and follows clean, modular practices with a responsive interface.

> ⚡ This project reuses and adapts a front-end template originally built for a POS system, redesigned specifically for attendance tracking in educational contexts.

---

## 🚀 Features

- ✅ Add and manage **Students**, **Teachers**, and **Courses**
- ✅ Mark and track **attendance** with status (Present/Absent)
- 📅 Support for **multi-day scheduling** (e.g., Mon/Wed/Fri)
- 📈 Visual dashboards for:
  - **Top students** with the highest attendance
  - **Gender distribution** (doughnut chart)
  - **Total attendance rate**
- 📊 **Daily summaries** and statistics
- 🔍 Filter by day, presence, and course

---

## 🧠 Technologies Used

- **Backend:** Django (Python)
- **Database:** SQLite (default) — easily swappable with PostgreSQL
- **Frontend:** HTML, CSS, JavaScript
- **Charts:** Chart.js for visualizations
- **Styling:** Custom + Bootstrap-inspired template reused from POS UI

---


## 🛠 Setup Instructions

1. **Clone the project:**
   ```bash
   git clone https://github.com/usmonaliyev-s/zedtrack.git
   cd zedtrack

2. **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
   
3. **Apply migrations:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
   
4. **Run the development server:**
    ```bash
   python manage.py runserver
5. Access ZedTrack:
   Open http://127.0.0.1:8000/ in your browser.

## 📌 Future Ideas (Planned Features)
- 🔮 Predictive attendance using ML (e.g., forecast student absenteeism)

## 🎓 Author
**Salohiddin Usmonaliyev**\
