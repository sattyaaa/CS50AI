from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")


# Basic knowledge
knowledge = And(
    # A can either be Knight or Knave not both
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),

    # B can either be Knight or Knave not both
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),

    # C can either be Knight or Knave not both
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave)),
)


# Puzzle 0
knowledge0 = And(
    knowledge,

    # A says "I am both a knight and a knave."
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave)))
)


# Puzzle 1
knowledge1 = And(
    knowledge,

    # A says "We are both knaves."
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave)))
    
    # B says nothing.   
)


# Puzzle 2
# A's statement: "We are the same kind"
A_statement = Or(
    And(AKnight, BKnight),
    And(AKnave, BKnave)
)

# B's statement: "We are of different kinds"
B_statement = Or(
    And(AKnight, BKnave),
    And(AKnave, BKnight)
)

knowledge2 = And(
    knowledge,

    # A says “We are the same kind.”
    Implication(AKnight, A_statement),
    Implication(AKnave, Not(A_statement)),

    # B says “We are of different kinds.”
    Implication(BKnight, B_statement),
    Implication(BKnave, Not(B_statement))
)


# Puzzle 3
knowledge3 = And(
    knowledge, 

    # A says either "I am a knight." or "I am a knave.", but you don't know which.
    Implication(AKnight, Or(AKnight, AKnave)),
    Implication(AKnave, Not(Or(AKnight, AKnave))),

    # B says "A said 'I am a knave'."
    Implication(BKnight, AKnave),
    Implication(BKnave, Not(AKnave)),

    # B says "C is a knave."
    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),

    # C says "A is a knight."
    Implication(CKnight, AKnight), 
    Implication(CKnave, Not(AKnight))
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
