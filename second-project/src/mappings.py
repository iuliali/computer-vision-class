from constants import Character, BARNEY, BETTY, FRED, WILMA


def mapping(character: Character):
    if character == Character.BARNEY:
        return BARNEY
    elif character == Character.BETTY:
        return BETTY
    elif character == Character.FRED:
        return FRED
    elif character == Character.WILMA:
        return WILMA
    else:
        return "unknown"


def reverse_mapping(character: str):
    if character == BARNEY:
        return Character.BARNEY
    elif character == BETTY:
        return Character.BETTY
    elif character == FRED:
        return Character.FRED
    elif character == WILMA:
        return Character.WILMA
    else:
        return 100
