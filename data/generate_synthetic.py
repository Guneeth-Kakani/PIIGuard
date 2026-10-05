"""
Synthetic PII test-case generator.

Design principle: separate SENTENCE STRUCTURE (templates) from ENTITY
CONTENT (Faker-generated values), and randomly combine them. This gives
combinatorial variety instead of N hand-written sentences.

Produces two disjoint files:
  - synthetic_dev.json    : used during development/iteration (Phases 2-4)
  - synthetic_heldout.json: generated with a different random seed and a
                            partially different template set; NOT looked
                            at until final evaluation (Phase 5). Treat it
                            as sealed the moment it's written.
"""

import json
import random
from faker import Faker

fake_en = Faker("en_US")
fake_in = Faker("en_IN")  # gives India-flavored names/addresses where supported

# ---------------------------------------------------------------------
# Templates: {SLOT} placeholders get filled with generated entity values.
# Kept deliberately varied in register/structure: formal, casual, code-like,
# mid-sentence, multi-entity, no-PII negatives.
# ---------------------------------------------------------------------

TEMPLATES = [
    "Please reach out to {PERSON} at {EMAIL} before end of day.",
    "{PERSON} called from {PHONE} regarding the {ORG} account.",
    "Ticket #4521: {PERSON} reported an issue, contact via {EMAIL}.",
    "Patient {PERSON} was diagnosed with {MEDICAL_CONDITION} on this visit.",
    "New employee {PERSON} will report to {PERSON2} starting Monday.",
    "Shipping to {ADDRESS}, attention {PERSON}.",
    "api_key = \"{API_KEY}\"  # do not commit this, {PERSON}",
    "Server log: request from {IP} authenticated as {PERSON}.",
    "{PERSON}'s card ending in {CREDIT_CARD} was declined at checkout.",
    "For verification, {PERSON} provided PAN {PAN} and phone {PHONE}.",
    "The meeting notes mention {ORG}'s Q3 targets but no personal data at all.",
    "{PERSON} and {PERSON2} are co-leading the {ORG} migration project.",
    "SSN on file: {SSN}, associated with applicant {PERSON}.",
    "Internal codename for the initiative is {CODENAME}, led by {PERSON}.",
    "This document contains no sensitive information, just general policy text.",
    "Emergency contact for {PERSON} is {PERSON2}, reachable at {PHONE}.",
]

CODENAMES = ["Project Nightingale", "Operation Bluebird", "Project Aurora",
             "Initiative Falcon", "Project Meridian", "Operation Silverline"]

MEDICAL_CONDITIONS = ["type 2 diabetes", "hypertension", "asthma",
                       "chronic migraine", "seasonal allergies", "anemia"]


def random_api_key():
    prefix = random.choice(["sk-ant-", "sk-", "AKIA"])
    length = 32 if prefix != "AKIA" else 16
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    body = "".join(random.choice(chars) for _ in range(length))
    return prefix + body if prefix != "AKIA" else prefix + body.upper()


def random_pan():
    letters = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
    digits = "".join(random.choice("0123456789") for _ in range(4))
    return f"{letters}{digits}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"


def random_aadhaar():
    """Shape-only: 12 digits grouped 4-4-4. Deliberately NOT checksum-valid
    or first-digit-restricted — matches the shape-only detector we test
    against (see detectors/presidio_baseline.py design note 2)."""
    digits = "".join(random.choice("0123456789") for _ in range(12))
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"


def gen_entity_values(rng_faker):
    return {
        "PERSON": rng_faker.name(),
        "PERSON2": rng_faker.name(),
        "EMAIL": rng_faker.email(),
        "PHONE": rng_faker.phone_number(),
        "ORG": rng_faker.company(),
        "ADDRESS": rng_faker.address().replace("\n", ", "),
        "API_KEY": random_api_key(),
        "IP": rng_faker.ipv4(),
        "CREDIT_CARD": rng_faker.credit_card_number(),
        "PAN": random_pan(),
        "AADHAAR": random_aadhaar(),
        "SSN": rng_faker.ssn(),
        "MEDICAL_CONDITION": random.choice(MEDICAL_CONDITIONS),
        "CODENAME": random.choice(CODENAMES),
    }


SLOT_TO_ENTITY_TYPE = {
    "PERSON": "PERSON", "PERSON2": "PERSON", "EMAIL": "EMAIL",
    "PHONE": "PHONE_NUMBER", "ORG": "ORGANIZATION", "ADDRESS": "ADDRESS",
    "API_KEY": "API_KEY", "IP": "IP_ADDRESS", "CREDIT_CARD": "CREDIT_CARD",
    "PAN": "PAN_NUMBER", "AADHAAR": "AADHAAR_LIKE", "SSN": "SSN",
    "MEDICAL_CONDITION": "MEDICAL_CONDITION", "CODENAME": "PROJECT_CODENAME",
}


def fill_template(template, faker_instance):
    values = gen_entity_values(faker_instance)
    text = template
    entities = []
    # Fill slots one at a time, tracking positions as we go, since each
    # substitution shifts subsequent offsets.
    while "{" in text:
        start_brace = text.index("{")
        end_brace = text.index("}", start_brace)
        slot = text[start_brace + 1:end_brace]
        value = values[slot]
        text = text[:start_brace] + value + text[end_brace + 1:]
        entities.append({
            "type": SLOT_TO_ENTITY_TYPE[slot],
            "text": value,
            "start": start_brace,
            "end": start_brace + len(value),
        })
    return text, entities


def generate_set(n, faker_instance, template_pool, seed):
    random.seed(seed)
    faker_instance.seed_instance(seed)
    cases = []
    for i in range(n):
        template = random.choice(template_pool)
        text, entities = fill_template(template, faker_instance)
        cases.append({
            "id": f"synth_{seed}_{i:04d}",
            "category": "synthetic",
            "text": text,
            "entities": entities,
        })
    return cases


def validate(cases):
    errors = []
    for case in cases:
        text = case["text"]
        for ent in case["entities"]:
            if text[ent["start"]:ent["end"]] != ent["text"]:
                errors.append(case["id"])
    return errors


if __name__ == "__main__":
    # DEV set: templates 0-13, seed 42, used freely during Phases 2-4
    dev_templates = TEMPLATES[:14]
    dev_cases = generate_set(600, fake_en, dev_templates, seed=42)
    dev_cases += generate_set(200, fake_in, dev_templates, seed=142)

    # HELD-OUT set: full template pool incl. ones dev set never sees (14, 15),
    # different seed. Do not open this file again until Phase 5.
    heldout_cases = generate_set(150, fake_en, TEMPLATES, seed=999)
    heldout_cases += generate_set(50, fake_in, TEMPLATES, seed=1999)

    for name, cases in [("dev", dev_cases), ("heldout", heldout_cases)]:
        errors = validate(cases)
        if errors:
            print(f"{name}: {len(errors)} index errors: {errors[:5]}")
            raise SystemExit(1)
        path = f"data/synthetic_{name}.json"
        with open(path, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"{name}: {len(cases)} cases validated and written to {path}")
