from tinkoff.invest.schemas import Quotation


def convert_to_money(value: Quotation) -> float:
    price = value.units + value.nano * 10**(-9)
    return price


def find_first(iterable, condition):
    try:
        return next(elem for elem in iterable if condition(elem))
    except StopIteration:
        return None
