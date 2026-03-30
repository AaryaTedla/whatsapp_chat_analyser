"""Generate realistic sample WhatsApp exports for testing."""
import random
from datetime import datetime, timedelta

SENDERS = ["Aarya", "Nikhil", "Pavi", "Rohit", "Sophia", "Aditya", "Maya", "Harsh"]
EMOTIONS = ["😀", "😂", "🎉", "🔥", "💯", "👍", "❤️", "😍", "😢", "😡", "🤤", "👌", "💪", "🚀"]
TOPICS = [
    "standup", "deployment", "api latency", "bug fix", "test case", "review", "merge",
    "sprint", "release", "production", "staging", "development", "deadline", "meeting",
    "lunch", "coffee", "gym", "movie", "game", "music", "travel", "weekend", "party",
    "weather", "traffic", "code review", "documentation", "optimization", "refactor"
]
ACTIONS = [
    "Can someone review", "Fixed the", "Deployed", "Just finished", "Working on",
    "Found a bug in", "Checking the", "Updating", "Created PR", "Merged", "Tested",
    "Need to fix", "Let's discuss", "Agreed about", "Disagree on", "I think we should",
    "Has anyone seen", "Where is", "When will we", "Looking good", "Updated the"
]
MEDIA = [
    "<image omitted>", "<video omitted>", "<audio omitted>", "<document omitted>"
]
LINKS = [
    "https://github.com/project/issues/42",
    "https://docs.example.com/api",
    "https://stackoverflow.com/questions/12345",
    "https://example.com/dashboard",
    "https://jira.company.com/browse/PROJ-123",
    "https://slack.com/archives/123",
    "https://github.com/pulls/999",
]

def generate_timestamp(base_date, offset_range):
    """Generate random timestamp."""
    days_offset = random.randint(0, offset_range)
    hour = random.randint(6, 23)  # 6 AM to 11 PM
    minute = random.randint(0, 59)
    dt = base_date + timedelta(days=days_offset, hours=hour, minutes=minute)
    return dt.strftime("%d/%m/%Y, %H:%M")

def generate_message():
    """Generate a realistic WhatsApp message."""
    msg_type = random.choice([
        "action", "action", "action",  # More likely to be regular messages
        "question", "response", "emoji", "link", "media", "hindi", "mixed"
    ])
    
    if msg_type == "action":
        action = random.choice(ACTIONS)
        topic = random.choice(TOPICS)
        return f"{action} {topic}?"
    elif msg_type == "question":
        return random.choice([
            "What do you think about this?",
            "Can you help me with this?",
            "Did you see the latest update?",
            "Are we on track?",
            "When is the deadline?",
            "Should we schedule a meeting?",
            "Is this working in your environment?",
        ])
    elif msg_type == "response":
        return random.choice([
            "Looks good!", "Agreed!", "Not sure yet.", "Let's discuss in standup.",
            "I'll check and get back to you.", "All green!", "Needs investigation.",
            "Thanks for the update!", "LGTM", "Good catch!", "Already done.",
            "On it!", "Will do.", "Sounds good to me.", "I think so too."
        ])
    elif msg_type == "emoji":
        emoji = random.choice(EMOTIONS)
        text = random.choice(["", "haha", "yay!", "oof", "nice", "yup"])
        return f"{text} {emoji}".strip()
    elif msg_type == "link":
        link = random.choice(LINKS)
        text = random.choice(["Check this out", "Found this helpful", "Updated:", "PR:", ""])
        return f"{text} {link}".strip() if text else link
    elif msg_type == "media":
        media = random.choice(MEDIA)
        text = random.choice(["", "Check this", "Latest screenshot", "Design mockup"])
        return f"{text} {media}".strip() if text else media
    elif msg_type == "hindi":
        hindi_phrases = [
            "Kya tum ye dekhe ho?", "Shukriya!", "Bilkul thik hai",
            "Main kya karu?", "Acha theek hai", "Yeh sahi hai na?", "Mujhe nahi pata"
        ]
        return random.choice(hindi_phrases)
    else:  # mixed
        topic = random.choice(TOPICS)
        emoji = random.choice(EMOTIONS)
        return f"Working on {topic} {emoji}"

def generate_export(filename, num_messages=5000, days_span=180):
    """Generate a sample WhatsApp export file."""
    base_date = datetime(2025, 9, 1)  # Start 6 months ago
    
    lines = []
    for _ in range(num_messages):
        timestamp = generate_timestamp(base_date, days_span)
        sender = random.choice(SENDERS)
        message = generate_message()
        
        # Format: DD/MM/YYYY, HH:MM - Sender: Message (expected by parser)
        line = f"{timestamp} - {sender}: {message}"
        lines.append(line)
    
    # Sort by date to be more realistic
    lines.sort()
    
    with open(f"/home/aaryatedla/lumira_prep/sample_exports/{filename}", "w") as f:
        f.write("\n".join(lines))
    
    print(f"✓ Generated {filename} with {num_messages} messages")

# Generate diverse test files
if __name__ == "__main__":
    generate_export("whatsapp_export_01_study_group.txt", num_messages=3500, days_span=120)
    generate_export("whatsapp_export_02_startup_team.txt", num_messages=5500, days_span=180)
    generate_export("whatsapp_export_03_family_trip.txt", num_messages=2200, days_span=45)
    generate_export("whatsapp_export_04_cricket_friends.txt", num_messages=4200, days_span=150)
    generate_export("whatsapp_export_05_hackathon.txt", num_messages=6800, days_span=200)
    generate_export("whatsapp_export_06_roommates.txt", num_messages=3800, days_span=100)
    generate_export("whatsapp_export_07_book_club.txt", num_messages=2500, days_span=80)
    generate_export("whatsapp_export_08_product_feedback.txt", num_messages=4100, days_span=140)
    generate_export("whatsapp_export_09_event_planning.txt", num_messages=5200, days_span=160)
    generate_export("whatsapp_export_10_support_ops.txt", num_messages=7500, days_span=210)
    
    print("\n✨ All test exports generated successfully!")
