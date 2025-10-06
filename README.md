# 🤖 AI Meeting Assistant

## 📘 Overview

The **AI Meeting Assistant** is an intelligent automation tool that helps users **schedule meetings**, **set reminders**, and **manage tasks** — all through **Telegram messages**.
It integrates **Telegram Bot**, **Todoist**, and **Google Calendar** to provide a seamless experience for managing meetings and daily productivity.

---

## 🚀 Features

* 📨 **Telegram Integration:** Receive and send commands/messages through a Telegram bot.
* 📅 **Calendar Scheduling:** Automatically schedules meetings on **Google Calendar**.
* ✅ **Task Management:** Creates and updates tasks in **Todoist** based on meeting details.
* 🔔 **Reminders:** Sends reminders or notifications before meetings.
* 🧠 **AI-Powered Understanding:** Uses NLP to interpret meeting-related commands like

  > “Schedule a meeting with John tomorrow at 3 PM”
  > “Add ‘Prepare presentation’ to my to-do list”

---

## 🧩 Tech Stack

* **Programming Language:** Python 🐍
* **APIs Used:**

  * Telegram Bot API
  * Todoist API
  * Google Calendar API
* **Libraries:**

  * `python-telegram-bot`
  * `google-api-python-client`
  * `todoist-api-python`
  * `datetime`, `os`, `dotenv`

---

## ⚙️ How It Works

1. **User sends a message** via Telegram (e.g., “Schedule meeting with Lata at 2 PM”).
2. The **Telegram Bot** receives and parses the command.
3. Depending on the command:

   * The meeting is added to **Google Calendar**.
   * A new task is created in **Todoist**.
4. The bot **confirms the action** back to the user via Telegram.
5. Optionally, reminders and updates are sent automatically.

---

## 🪄 Example Commands

| Command                                     | Action                                 |
| ------------------------------------------- | -------------------------------------- |
| `/schedule meeting with Alex tomorrow 11am` | Adds meeting to Google Calendar        |
| `/add task Submit report`                   | Adds a new Todoist task                |
| `/show tasks`                               | Displays current Todoist tasks         |
| `/next meeting`                             | Shows upcoming Google Calendar meeting |

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/AI_Meeting_Assistant.git
cd AI_Meeting_Assistant
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a `.env` file in the project root and add:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TODOIST_API_KEY=your_todoist_api_key
GOOGLE_CREDENTIALS_PATH=path_to_google_credentials.json
```

### 5️⃣ Run the Bot

```bash
streamlit run meeting_agent.py
```

---

## 🧠 Future Enhancements

* Integration with **Microsoft Teams** or **Zoom** for automatic meeting links
* **Voice command support** for scheduling via speech
* **Smart conflict detection** for overlapping meetings
* **Automatic meeting summaries** using AI transcription


