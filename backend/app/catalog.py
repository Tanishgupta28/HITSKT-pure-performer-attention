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
           "level": "Foundations", "concepts": CONCEPTS, "content_version": "math-original-v2"}

# Added lessons and items preserve the IDs, grading and version of all v1 items.
LESSON_DETAILS = {
    "fractions": {
        "steps": ["Find a common denominator.", "Rewrite each fraction with that denominator.", "Add or subtract the numerators, then simplify."],
        "worked_example": "3/4 − 1/6 = 9/12 − 2/12 = 7/12. To find 3/4 of 20, calculate 20 × 3/4 = 15.",
        "common_mistake": "Do not add the denominators: 1/2 + 1/2 is 1, not 2/4.",
        "application": "Sharing food, measuring ingredients and dividing quantities fairly."},
    "percentages": {
        "steps": ["Write the percentage as a fraction out of 100.", "Multiply by the original quantity.", "For discounts subtract this amount; for increases add it."],
        "worked_example": "A ₹200 book has a 15% discount. The discount is 200 × 15/100 = ₹30, so the sale price is ₹170.",
        "common_mistake": "The discount amount and the final price are different quantities. Check what the question asks.",
        "application": "Shopping discounts, savings goals and changes in everyday prices."},
    "equations": {
        "steps": ["Write an equation for the unknown quantity.", "Undo addition or subtraction on both sides.", "Undo multiplication or division, then check your answer."],
        "worked_example": "Three identical notebooks and a ₹10 pen cost ₹70. If each notebook costs x, then 3x + 10 = 70, so 3x = 60 and x = ₹20.",
        "common_mistake": "An operation on one side must also be applied to the other side.",
        "application": "Finding unknown prices, splitting bills and planning a budget."},
    "geometry": {
        "steps": ["Decide whether the question asks for space (area) or an edge (perimeter).", "Choose the formula for the shape.", "Substitute the dimensions and use the correct units."],
        "worked_example": "A triangle with base 8 cm and height 5 cm has area 1/2 × 8 × 5 = 20 square cm. Its base and height alone do not determine its perimeter.",
        "common_mistake": "Area uses square units. Perimeter uses length units. A triangle's height must be perpendicular to its base.",
        "application": "Buying fencing, planning a room and estimating the amount of flooring."},
}
for concept in CONCEPTS:
    concept.update(LESSON_DETAILS[concept["id"]])


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
    for level in (1, 2, 3):
        for variant in range(17, 33):
            n = variant - 16
            denominator = n + 4
            if level < 3:
                left, right = Fraction(denominator - 1, denominator), Fraction(1, denominator * level)
                result = left - right
                prompt = f"What is {left} − {right}?"
                explanation = f"Use a common denominator of {denominator * level}. Subtract the numerators and simplify: {left} − {right} = {result}."
            else:
                quantity = denominator * (n + 2)
                left = Fraction(3, denominator)
                result = left * quantity
                prompt = f"A class has {quantity} students. {left} of them walk to school. How many students walk?"
                explanation = f"Multiply the class size by the fraction: {quantity} × {left} = {result} students."
            bank.append(_question("fractions", level, variant, prompt, result,
                                  [result + 1, result / 2, result + 2], explanation))
            base, pct = (n + 4) * 100, (10, 15, 25)[level - 1]
            change = base * pct // 100
            if level == 1:
                answer = base - change
                prompt = f"A bag costs ₹{base}. It has a {pct}% discount. What is the sale price in rupees?"
                explanation = f"The discount is {base} × {pct}/100 = ₹{change}. Subtract it: {base} − {change} = ₹{answer}."
            elif level == 2:
                answer = base + change
                prompt = f"A monthly savings target of ₹{base} increases by {pct}%. What is the new target in rupees?"
                explanation = f"The increase is {base} × {pct}/100 = ₹{change}. Add it: {base} + {change} = ₹{answer}."
            else:
                answer = pct
                prompt = f"A club grows from {base} members to {base + change}. What is the percentage increase?"
                explanation = f"Increase = {change}. Divide by the original {base}, then multiply by 100: {change}/{base} × 100 = {pct}%."
            bank.append(_question("percentages", level, variant, prompt, answer,
                                  [change, answer + 5, answer * 2], explanation))
            x, multiplier, offset = n + 3, level + 2, n + 5
            if level < 3:
                total = multiplier * x + offset
                prompt = f"{multiplier} identical notebooks and a ₹{offset} pen cost ₹{total}. What is the price of one notebook in rupees?"
                explanation = f"Let x be one notebook's price. {multiplier}x + {offset} = {total}. Subtract {offset}, then divide by {multiplier}: x = {x}."
            else:
                total = multiplier * (x + offset)
                prompt = f"Solve for x: {multiplier}(x + {offset}) = {total}."
                explanation = f"Divide both sides by {multiplier}: x + {offset} = {x + offset}. Subtract {offset}: x = {x}."
            bank.append(_question("equations", level, variant, prompt, x,
                                  [x + offset, x - 1, x * multiplier], explanation))
            side = n + 4
            if level == 1:
                answer = 4 * side
                prompt = f"A square garden has sides of {side} m. How many metres of fencing enclose it?"
                explanation = f"A square has four equal sides. Perimeter = 4 × {side} = {answer} m."
            elif level == 2:
                answer = side * side
                prompt = f"A square tile has sides of {side} cm. What is its area in square cm?"
                explanation = f"Square area = side × side = {side} × {side} = {answer} square cm."
            else:
                base, height = 2 * side, n + 3
                answer = base * height // 2
                prompt = f"A triangle has base {base} cm and perpendicular height {height} cm. What is its area in square cm?"
                explanation = f"Triangle area = 1/2 × base × height = 1/2 × {base} × {height} = {answer} square cm."
            bank.append(_question("geometry", level, variant, prompt, answer,
                                  [answer + side, answer * 2, answer - 1], explanation))
            for question in bank[-4:]:
                question["content_version"] = "math-original-v2"
    return {question["id"]: question for question in bank}


BANK = build_bank()


def public_question(question):
    return {key: value for key, value in question.items() if key not in {"answer_index", "explanation"}}
