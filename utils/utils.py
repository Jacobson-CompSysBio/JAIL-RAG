def get_allowed_values_as_str(x: list[str]) -> str:
  if len(x) == 1:
    return f'{x[0]}'
  else:
    return f"{', '.join([str(a) for a in x[:-1]])} or {x[-1]}"