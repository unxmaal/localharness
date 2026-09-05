"""Reduce a Claude response to speakable prose, keeping any closing question intact.

Reads the raw message on stdin, writes JSON on stdout:

    {"body": "...prose to summarize...", "question": "...verbatim or empty..."}

Why the question is split out rather than summarized with everything else: a
small instruct model paraphrasing "Want me to wire this up next?" produces
"Next, I'll wire up the voice layer" often enough to matter. That converts a
question into a commitment the speaker never made. Observed repeatedly with
Qwen2.5-1.5B, and non-deterministically, which is worse than consistently wrong.

The closing question is the highest-stakes sentence in the message, so it is
never sent to the model at all. It is extracted here and reattached verbatim
after summarization.
"""
import json
import re
import sys

t = sys.stdin.read()

# Code and paths are DROPPED, not replaced with a marker. A "(code)" or "a path"
# placeholder is an invitation: the summarizer writes "see the code for more
# details" about code the listener cannot see, which is a fabrication.
t = re.sub(r"```.*?```", " ", t, flags=re.S)          # fenced blocks
t = re.sub(r"`[^`]+`", " ", t)                        # inline code
t = re.sub(r"^\s*\|.*\|\s*$", "", t, flags=re.M)      # table rows
t = re.sub(r"^\s*[-=|:+]{3,}\s*$", "", t, flags=re.M) # rules, table separators
t = re.sub(r"https?://\S+", " ", t)
t = re.sub(r"(?:/[\w.\-]+){2,}", " ", t)              # unix paths
t = re.sub(r"[#*_>`]", "", t)                         # markdown punctuation

# Dropping content leaves stranded referring phrases ("See ."). Remove the
# fragments that pointed at what was just deleted.
t = re.sub(r"(?i)\b(see|check|from|in|at|run)\s*[:,]?\s*(?=[.;!?])", "", t)
t = re.sub(r"\s+([.,;!?])", r"\1", t)
t = re.sub(r"([.;!?])(\s*\1)+", r"\1", t)
t = re.sub(r"\s+", " ", t).strip()

question = ""
sentences = re.split(r"(?<=[.!?])\s+", t)
if sentences and sentences[-1].rstrip().endswith("?"):
    question = sentences.pop().strip()
    t = " ".join(sentences).strip()

print(json.dumps({"body": t[:6000], "question": question}))
