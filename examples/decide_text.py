"""Decisions about text and JSON: route support tickets with questions from a YAML file.

python examples/decide_text.py --model d1-omni-600m --runtime ollama
"""

import argparse
from pathlib import Path

from s1ght import S1GHT

TICKETS = [
    "I was charged twice this month, please refund one of them.",
    "The app crashes every time I open the camera screen.",
    {"subject": "Unknown login", "body": "Someone in another country signed in to my account and changed my email."},
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="d1-omni-600m")
ap.add_argument("--runtime", default=None)
args = ap.parse_args()

questions = Path(__file__).with_name("ticket.yaml")
with S1GHT(args.model, runtime=args.runtime) as model:
    for ticket in TICKETS:
        a = model.decide(ticket, questions).answers
        text = ticket if isinstance(ticket, str) else ticket["subject"]
        print(f"{text[:60]:60s}  refund={a['refund'].noul:.2f}  team={a['team'].choice:9s}  "
              f"urgency={a['urgency'].score:.1f}/2")  # fmt: skip
