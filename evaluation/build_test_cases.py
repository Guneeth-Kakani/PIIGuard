"""
Builds evaluation/test_cases.json from human-readable case definitions.

Instead of hand-typing start/end character indices (error-prone, as we
just learned), each case is defined as: the text, plus a list of
(entity_type, exact_substring) pairs. This script finds the substring
in the text and computes indices automatically.

If a substring appears more than once in a text, pass an `occurrence`
index (0-based) as a third tuple element to disambiguate.
"""

import json

# Each case: (id, category, text, [(entity_type, substring[, occurrence]), ...])
RAW_CASES = [
    ("clean_001", "clean_single_entity",
     "Please contact John Smith regarding the account.",
     [("PERSON", "John Smith")]),

    ("clean_002", "clean_single_entity",
     "You can reach me at priya.sharma@techcorp.com for any questions.",
     [("EMAIL", "priya.sharma@techcorp.com")]),

    ("multi_001", "multiple_entities_same_sentence",
     "Rahul Verma's phone number is +91-98765-43210 and his email is rahul.v@company.in.",
     [("PERSON", "Rahul Verma"),
      ("PHONE_NUMBER", "+91-98765-43210"),
      ("EMAIL", "rahul.v@company.in")]),

    ("multi_002", "multiple_entities_same_sentence",
     "Dr. Anjali Mehta prescribed metformin for Rohan's type 2 diabetes at Apollo Hospital.",
     [("PERSON", "Anjali Mehta"),
      ("PERSON", "Rohan"),
      ("MEDICAL_CONDITION", "type 2 diabetes"),
      ("ORGANIZATION", "Apollo Hospital")]),

    ("code_001", "pii_in_code_block",
     'Here\'s my config:\napi_key = "sk-ant-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"\nemail = "dev@startup.io"',
     [("API_KEY", "sk-ant-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"),
      ("EMAIL", "dev@startup.io")]),

    ("code_002", "pii_in_code_block",
     "curl -H 'Authorization: Bearer AKIAIOSFODNN7EXAMPLE' https://api.internal.com/users?ip=192.168.1.45",
     [("API_KEY", "AKIAIOSFODNN7EXAMPLE"),
      ("IP_ADDRESS", "192.168.1.45")]),

    ("ambig_001", "ambiguous_names",
     "Grace decided to say grace before the meal with Will and Rose.",
     [("PERSON", "Grace", 0),   # the capitalized name, first occurrence
      ("PERSON", "Will"),
      ("PERSON", "Rose")]),

    ("ambig_002", "ambiguous_names",
     "May will join the June project after March completes her review.",
     [("PERSON", "May"),
      ("PERSON", "March")]),

    ("structured_001", "structured_odd_formats",
     "Card number: 4111 1111 1111 1111, expiry 08/27, SSN 123 45 6789.",
     [("CREDIT_CARD", "4111 1111 1111 1111"),
      ("SSN", "123 45 6789")]),

    ("structured_002", "structured_odd_formats",
     "Call me at 98765 43210 or on my landline 040-2345-6789.",
     [("PHONE_NUMBER", "98765 43210"),
      ("PHONE_NUMBER", "040-2345-6789")]),

    ("india_001", "india_specific",
     "My PAN is ABCDE1234F and Aadhaar number is 1234 5678 9012.",
     [("PAN_NUMBER", "ABCDE1234F"),
      ("AADHAAR_LIKE", "1234 5678 9012")]),

    ("india_002", "india_specific",
     "Send the offer letter to Kavya Reddy at Flat 302, Prestige Towers, Hyderabad, and cc her manager Suresh.",
     [("PERSON", "Kavya Reddy"),
      ("ADDRESS", "Flat 302, Prestige Towers, Hyderabad"),
      ("PERSON", "Suresh")]),

    ("negative_001", "no_pii_present",
     "The quarterly report shows a 12% increase in revenue compared to last year.",
     []),

    ("negative_002", "no_pii_present",
     "Please summarize the key findings from the attached research paper on climate change.",
     []),

    ("nested_001", "nested_adjacent_entities",
     "New hire Ananya Rao (ananya.rao@corp.com, +91-99887-76655) starts Monday, reporting to Vikram Singh.",
     [("PERSON", "Ananya Rao"),
      ("EMAIL", "ananya.rao@corp.com"),
      ("PHONE_NUMBER", "+91-99887-76655"),
      ("PERSON", "Vikram Singh")]),

    ("zero_shot_001", "project_codename",
     "Project Nightingale's launch has been delayed until the Q3 review with stakeholders.",
     [("PROJECT_CODENAME", "Project Nightingale")]),
]


def build():
    cases = []
    for case_id, category, text, entity_defs in RAW_CASES:
        entities = []
        for entity_def in entity_defs:
            if len(entity_def) == 3:
                etype, substring, occurrence = entity_def
            else:
                etype, substring = entity_def
                occurrence = 0

            # find the nth occurrence of substring in text
            start = -1
            search_from = 0
            for _ in range(occurrence + 1):
                start = text.find(substring, search_from)
                if start == -1:
                    raise ValueError(
                        f"[{case_id}] substring {substring!r} (occurrence {occurrence}) "
                        f"not found in text: {text!r}"
                    )
                search_from = start + 1
            end = start + len(substring)
            entities.append({
                "type": etype,
                "text": substring,
                "start": start,
                "end": end,
            })
        cases.append({
            "id": case_id,
            "category": category,
            "text": text,
            "entities": entities,
        })
    return cases


def validate(cases):
    errors = []
    for case in cases:
        text = case["text"]
        for ent in case["entities"]:
            extracted = text[ent["start"]:ent["end"]]
            if extracted != ent["text"]:
                errors.append((case["id"], ent["type"], ent["text"], extracted))
    return errors


if __name__ == "__main__":
    cases = build()
    errors = validate(cases)
    if errors:
        print(f"{len(errors)} index mismatches found:")
        for e in errors:
            print(e)
        raise SystemExit(1)

    total_entities = sum(len(c["entities"]) for c in cases)
    print(f"All {total_entities} entity spans across {len(cases)} test cases verified correct.")

    with open("evaluation/test_cases.json", "w") as f:
        json.dump(cases, f, indent=2)
    print("Wrote evaluation/test_cases.json")
