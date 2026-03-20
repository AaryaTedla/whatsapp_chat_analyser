import re
from datetime import datetime
from typing import Dict, List, Optional


# Example WhatsApp lines (varies by locale/export):
# 20/03/2020, 09:15 - John Doe: Hello
# 14/10/25, 2:31 pm - Pavithraa: My data is not working
# 20/03/2020, 09:15 - Messages to this group are now secured with end-to-end encryption.
_LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})\s*(?P<ampm>am|pm)?\s*-\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_SENDER_SPLIT_RE = re.compile(r"^(?P<sender>.+?):\s(?P<message>.*)$")

# WhatsApp uses dd/mm/yyyy (or dd/mm/yy) and 24h times.
_DATETIME_FORMATS = ("%d/%m/%Y %H:%M", "%d/%m/%y %H:%M")


def _to_24h(time_str: str, ampm: Optional[str]) -> str:
    """Convert '2:31' + 'pm' to '14:31'."""
    if not ampm:
        return time_str
    h, _, m = time_str.partition(":")
    hi = int(h) % 12
    if ampm.lower() == "pm":
        hi += 12
    return f"{hi:02d}:{m}"


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    dt_str = f"{date_str} {time_str}"
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    # Fallback: try two-digit year coercion.
    try:
        d, m, y = date_str.split("/")
        if len(y) == 2:
            y = str(2000 + int(y))
            dt_str2 = f"{d}/{m}/{y} {time_str}"
            return datetime.strptime(dt_str2, "%d/%m/%Y %H:%M")
    except Exception:
        pass
    return None


def parse_whatsapp_export(text: str) -> List[Dict]:
    """
    Parse a WhatsApp exported chat text (.txt).

    Returns a list of dicts with keys: date, time, datetime, sender, message.
    """
    lines = text.splitlines()

    results: List[Dict] = []
    current: Optional[Dict] = None

    for raw_line in lines:
        line = raw_line.strip("\ufeff").rstrip()
        if not line:
            continue

        m = _LINE_RE.match(line)
        if not m:
            # Continuation of previous message (multi-line messages).
            if current is not None:
                current["message"] = f"{current['message']}\n{line}".strip()
            continue

        # Close out previous message, if any.
        if current is not None:
            results.append(current)

        date_str = m.group("date")
        time_str = m.group("time")
        ampm = m.group("ampm")
        time_24h = _to_24h(time_str, ampm)
        dt = _parse_datetime(date_str, time_24h)
        rest = m.group("rest").strip()

        sender = "Unknown"
        message = rest

        # Most common: "<sender>: <message>"
        sms = _SENDER_SPLIT_RE.match(rest)
        if sms:
            sender = sms.group("sender").strip()
            message = sms.group("message").strip()
        else:
            # System messages usually don't have a sender/name + colon.
            sender = "System"
            message = rest

        current = {
            "date": date_str,
            "time": time_str,
            "datetime": dt.isoformat() if dt else None,
            "sender": sender,
            "message": message,
        }

    if current is not None:
        results.append(current)

    # Normalize datetime to actual datetime strings or None.
    return [r for r in results if r.get("message") is not None]

