"""
Phase 4 corpus generator — larger, more structurally diverse than the
Phase 2/3 eval sets (synthetic_dev.json / synthetic_heldout.json), which
stay untouched and frozen. This corpus exists to be auto-labeled by the
ensemble detector, not manually verified — but since entities are still
injected via known slot values, we keep that ground truth around
separately as `injected_entities`, purely to measure how much the
ensemble's auto-labels agree/disagree with known truth (a free pseudo-
label quality check, not a training signal).
"""

import json
import random
from faker import Faker

# Reuse the entity-value generators already built and tested in Phase 2/3
from generate_synthetic import (
    gen_entity_values, SLOT_TO_ENTITY_TYPE, fill_template, validate,
)

fake_en = Faker("en_US")
fake_in = Faker("en_IN")
fake_gb = Faker("en_GB")

# ---------------------------------------------------------------------
# Much larger template pool than Phase 2/3's 16. Deliberately varied:
#   - multi-sentence "documents" (real prompts rarely arrive as one line)
#   - casual chat / Slack-style register
#   - code + log snippets (where PII often leaks unnoticed)
#   - quoted third-party speech (tests boundary handling)
#   - hard negatives: PII-shaped but NOT PII (phone-shaped order numbers,
#     capitalized non-names) to stress-test false-positive rate
#   - entity-free text on varied topics, so the model also learns "O"
#     (no entity) doesn't mean "text is short/simple"
# ---------------------------------------------------------------------

PHASE4_TEMPLATES = [
    # --- multi-sentence documents ---
    "Hi team, quick update. {PERSON} joined us from {ORG} this week. "
    "Please cc {PERSON2} on the onboarding email at {EMAIL}.",

    "Incident report: {PERSON} reported unauthorized access from IP {IP}. "
    "We've reset the credentials associated with {EMAIL} as a precaution.",

    "Patient intake notes: {PERSON}, contact {PHONE}. Presenting with "
    "{MEDICAL_CONDITION}. Referred by Dr. {PERSON2} at {ORG}.",

    "Meeting recap — attendees: {PERSON}, {PERSON2}. Action item: {PERSON} "
    "to follow up with finance regarding the {ORG} invoice by Friday.",

    # --- casual / chat register ---
    "hey can u send that to {EMAIL} instead, {PERSON} keeps missing it lol",
    "just talked to {PERSON} on {PHONE}, they're good with the {ORG} deal",
    "lol {PERSON} really said that in front of {PERSON2} 💀",

    # --- code / log snippets ---
    "// TODO: remove hardcoded key before prod\nAUTH_KEY = \"{API_KEY}\"\n"
    "// contact {PERSON} if this breaks",
    "[ERROR] 2024-01-12 auth failed for user session from {IP}, "
    "flagged account owner {PERSON} ({EMAIL})",
    "config.yaml:\n  admin_email: {EMAIL}\n  api_key: {API_KEY}\n  "
    "owner: {PERSON}",

    # --- quoted third-party speech ---
    "{PERSON} said, \"just email me at {EMAIL} and I'll handle it\" "
    "during the call with {ORG}.",
    "According to {PERSON2}, the codename for the launch is {CODENAME}, "
    "not to be shared outside the {ORG} team.",

    # --- forms / structured-ish free text ---
    "Applicant: {PERSON}\nPAN: {PAN}\nPhone: {PHONE}\nAddress: {ADDRESS}",
    "Refund request from {PERSON} ({EMAIL}) for card ending in "
    "{CREDIT_CARD}, order shipped to {ADDRESS}.",
    "KYC verification pending for {PERSON}. Aadhaar: {AADHAAR}, PAN: {PAN}.",
    "Identity confirmed for {PERSON} using Aadhaar {AADHAAR} and phone {PHONE}.",

    # --- SSN and CREDIT_CARD templates (added after training revealed a
    # class-imbalance bug: the original corpus had ZERO SSN templates and
    # only ONE CREDIT_CARD template, vs. 5+ templates for PHONE_NUMBER.
    # The model learned to default every digit-grouped string it saw to
    # PHONE_NUMBER, since it had essentially no other labeled examples of
    # similarly-shaped entities to learn from. These templates bring SSN
    # and CREDIT_CARD up to roughly the same frequency as PAN/AADHAAR.) ---
    "SSN on file: {SSN}, associated with applicant {PERSON}.",
    "Please verify your identity using SSN {SSN} before proceeding, {PERSON}.",
    "{PERSON}'s employment record lists SSN {SSN} for tax purposes.",
    "Payment declined for card {CREDIT_CARD} belonging to {PERSON}.",
    "{PERSON} updated their billing info, new card ending in {CREDIT_CARD}.",
    "Refund issued to card {CREDIT_CARD} for order placed by {PERSON}.",

    # --- one additional MEDICAL_CONDITION template — the original corpus
    # only had one (236 examples, the lowest non-zero count), worth a
    # modest bump alongside the SSN/CREDIT_CARD fix. ---
    "{PERSON} scheduled a follow-up appointment regarding {MEDICAL_CONDITION}.",

    # --- hard negatives: PII-shaped but not actually PII ---
    "Order #4471-8823-1092 shipped yesterday, tracking updates hourly.",
    "The API rate limit is 1000 requests per 60 seconds per key.",
    "Room 204B is booked from 14:00 to 16:30 for the workshop.",
    "Q3 revenue grew 12% to $4.2M, driven mostly by North region sales.",
    "Version 2.14.0 fixes a memory leak reported in issue #881.",

    # --- entity-free, varied topics (no PII at all) ---
    "The migration to the new data warehouse is expected to take three weeks.",
    "Please review the attached style guide before submitting your draft.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The conference has been moved to the downtown convention center.",

    # --- nested / adjacent, harder boundary cases ---
    "{PERSON}'s manager, {PERSON2}, approved the request from {ORG} "
    "on behalf of the {CODENAME} initiative.",
    "Emergency contact: {PERSON} ({PHONE}), alternate: {PERSON2} ({EMAIL}).",
]

SLOT_TO_ENTITY_TYPE.setdefault("ADDRESS", "ADDRESS")  # already present, safe no-op


def generate_phase4_corpus(n_per_locale, seed):
    random.seed(seed)
    cases = []
    locale_fakers = [fake_en, fake_in, fake_gb]
    i = 0
    for faker_instance in locale_fakers:
        faker_instance.seed_instance(seed)
        for _ in range(n_per_locale):
            template = random.choice(PHASE4_TEMPLATES)
            text, entities = fill_template(template, faker_instance)
            cases.append({
                "id": f"phase4_{seed}_{i:05d}",
                "text": text,
                "injected_entities": entities,  # NOT used for training directly
            })
            i += 1
    return cases


if __name__ == "__main__":
    corpus = generate_phase4_corpus(n_per_locale=2000, seed=7)  # ~6000 total

    errors = []
    for case in corpus:
        for ent in case["injected_entities"]:
            if case["text"][ent["start"]:ent["end"]] != ent["text"]:
                errors.append(case["id"])
    if errors:
        print(f"{len(errors)} index errors: {errors[:5]}")
        raise SystemExit(1)

    with open("data/phase4_corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Wrote {len(corpus)} cases to data/phase4_corpus.json")
