"""ISO-3166 alpha-2 <-> the country names used by powerplantmatching."""

NAME = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus",
    "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France",
    "DE": "Germany", "GR": "Greece", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia", "SI": "Slovenia",
    "ES": "Spain", "SE": "Sweden", "NO": "Norway", "CH": "Switzerland", "AL": "Albania",
    "BA": "Bosnia and Herzegovina", "ME": "Montenegro", "MK": "North Macedonia",
    "RS": "Serbia", "XK": "Kosovo", "GB": "United Kingdom", "UA": "Ukraine", "MD": "Moldova",
}
ISO = {v: k for k, v in NAME.items()}
# Eurostat uses EL for Greece (and UK for the United Kingdom).
EUROSTAT = {k: ("EL" if k == "GR" else k) for k in NAME}
