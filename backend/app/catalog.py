"""Original, deterministic practice content. No benchmark IDs are invented."""
from fractions import Fraction
from random import Random


CONCEPTS = [
    {"id": "fractions", "name": "Fractions", "description": "Add and compare parts of a whole.",
     "prerequisites": [], "lesson": "Use a common denominator before adding fractions. For example, 1/3 + 1/6 = 2/6 + 1/6 = 1/2."},
    {"id": "percentages", "name": "Percentages", "description": "Connect fractions, ratios and percentages.",
     "prerequisites": ["fractions"], "lesson": "A percentage means out of 100. To find 15% of 80, multiply 80 by 15/100 to get 12."},
    {"id": "equations", "name": "Linear equations", "description": "Find unknown values one step at a time.",
     "prerequisites": [], "lesson": "Keep both sides balanced. In 3x + 4 = 19, subtract 4 from each side, then divide by 3. So x = 5."},
    {"id": "geometry", "name": "Area & perimeter", "description": "Reason about the space and edges of shapes.",
     "prerequisites": ["equations"], "lesson": "Rectangle area = length × width. Perimeter = 2 × (length + width). Area uses square units; perimeter uses length units."},
]
SUBJECT = {"id": "mathematics", "name": "Mathematics", "subtitle": "Build a stronger foundation, one concept at a time.",
           "level": "Foundations", "concepts": CONCEPTS, "content_version": "math-original-v1"}


def _question(concept, level, variant, prompt, answer, distractors, explanation):
    choices = list(dict.fromkeys([str(answer), *map(str, distractors)]))
    extra = 1
    while len(choices) < 4:
        value = str(extra + 103)
        if value not in choices:
            choices.append(value)
        extra += 1
    choices = choices[:4]
    Random(f"{concept}:{level}:{variant}").shuffle(choices)
    return {"id": f"{concept}-{level}-{variant}", "subject_id": "mathematics", "concept_id": concept,
            "difficulty": level, "prompt": prompt, "choices": choices, "answer_index": choices.index(str(answer)),
            "explanation": explanation, "content_version": "math-original-v1"}


def build_bank():
    bank = []
    for level in (1, 2, 3):
        for variant in range(1, 17):
            denominator = level * 3 + variant % 5 + 2
            a, b = variant % 3 + 1, level + 1
            result = Fraction(a, denominator) + Fraction(b, denominator * (1 if level == 1 else 2))
            d2 = denominator * (1 if level == 1 else 2)
            bank.append(_question("fractions", level, variant, f"What is {a}/{denominator} + {b}/{d2}?", result,
                                  [result + 1, result / 2, result + Fraction(1, denominator)],
                                  f"Use a common denominator: {a}/{denominator} + {b}/{d2} = {result}. Reduce the fraction if possible."))
            base, pct = (variant + 3) * 20, (5, 15, 35)[level - 1]
            answer = base * pct // 100
            bank.append(_question("percentages", level, variant, f"What is {pct}% of {base}?", answer,
                                  [answer + 5, answer * 2, answer + pct], f"Multiply {base} × {pct}/100 = {answer}."))
            x, coefficient, offset = variant + 1, level + 1, variant + level
            total = coefficient * x + offset
            bank.append(_question("equations", level, variant, f"Solve for x: {coefficient}x + {offset} = {total}.", x,
                                  [x + coefficient, x - 1, total - offset],
                                  f"Subtract {offset} from both sides: {coefficient}x = {total - offset}. Divide by {coefficient}: x = {x}."))
            width, length = variant + 2, variant + level + 3
            perimeter = level == 2
            answer = 2 * (width + length) if perimeter else width * length
            operation = "perimeter" if perimeter else "area"
            bank.append(_question("geometry", level, variant,
                                  f"A rectangle is {length} cm long and {width} cm wide. What is its {operation} in {'cm' if perimeter else 'square cm'}?",
                                  answer, [answer + width, answer + length, answer - 1],
                                  f"{'Perimeter = 2 × (length + width)' if perimeter else 'Area = length × width'} = {answer}."))
    return {question["id"]: question for question in bank}


BANK = build_bank()


def public_question(question):
    return {key: value for key, value in question.items() if key not in {"answer_index", "explanation"}}
