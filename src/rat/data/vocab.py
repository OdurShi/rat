"""Word lists of the scene generator.  All entries are single tokens without spaces so that
answers that name a category, a row or a field value are compared by exact match."""

from __future__ import annotations

CATEGORY_SETS: dict[str, tuple[str, ...]] = {
    "month": (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ),
    "region": (
        "North",
        "South",
        "East",
        "West",
        "Central",
        "Coastal",
        "Inland",
        "Metro",
        "Rural",
    ),
    "product": (
        "Alpha",
        "Bravo",
        "Cobalt",
        "Delta",
        "Ember",
        "Falcon",
        "Garnet",
        "Harbor",
        "Indigo",
        "Jasper",
        "Kestrel",
        "Lumen",
    ),
    "team": (
        "Otters",
        "Hawks",
        "Lynx",
        "Bison",
        "Cranes",
        "Foxes",
        "Herons",
        "Wolves",
        "Ravens",
        "Pumas",
    ),
    "city": (
        "Avon",
        "Belmont",
        "Carver",
        "Dalton",
        "Easton",
        "Fairview",
        "Granby",
        "Hollis",
        "Irving",
        "Jordan",
        "Kenmore",
        "Linden",
    ),
    "department": (
        "Sales",
        "Support",
        "Design",
        "Finance",
        "Legal",
        "Research",
        "Logistics",
        "Quality",
    ),
}

#: Singular noun used in questions for each category set.
CATEGORY_NOUN: dict[str, str] = {
    "month": "month",
    "region": "region",
    "product": "product",
    "team": "team",
    "city": "city",
    "department": "department",
}

CHART_TITLES: tuple[str, ...] = (
    "Units sold",
    "Monthly revenue",
    "Support tickets",
    "Average score",
    "Visitors",
    "Orders shipped",
    "Energy use",
    "Sign-ups",
    "Budget spent",
    "Response time",
)

TABLE_TITLES: tuple[str, ...] = (
    "Quarterly summary",
    "Inventory report",
    "Season results",
    "Survey results",
    "Regional overview",
    "Cost breakdown",
    "Production log",
    "Attendance record",
)

TABLE_COLUMNS: tuple[str, ...] = (
    "Sales",
    "Units",
    "Cost",
    "Score",
    "Orders",
    "Stock",
    "Visits",
    "Hours",
    "Points",
    "Returns",
    "Output",
    "Profit",
)

FIRST_NAMES: tuple[str, ...] = (
    "Mara",
    "Theo",
    "Ines",
    "Jonas",
    "Leila",
    "Oskar",
    "Priya",
    "Rafael",
    "Selma",
    "Tariq",
    "Uma",
    "Viktor",
    "Wren",
    "Yusuf",
    "Zora",
    "Anders",
)

LAST_NAMES: tuple[str, ...] = (
    "Albrecht",
    "Bianchi",
    "Castell",
    "Dvorak",
    "Eklund",
    "Farrow",
    "Grieg",
    "Hartono",
    "Ivers",
    "Jansen",
    "Kovacs",
    "Lindqvist",
    "Moreau",
    "Novak",
    "Okafor",
    "Petrov",
)

RECEIPT_ITEMS: tuple[str, ...] = (
    "Coffee",
    "Bagel",
    "Notebook",
    "Batteries",
    "Sandwich",
    "Tea",
    "Pencils",
    "Juice",
    "Stamps",
    "Soup",
    "Muffin",
    "Charger",
    "Salad",
    "Folder",
    "Candle",
    "Tape",
)

SHOP_NAMES: tuple[str, ...] = (
    "Corner Market",
    "Harbor Cafe",
    "Pine Street Books",
    "Station Kiosk",
    "Maple Grocer",
    "Depot Supplies",
)

STATUS_WORDS: tuple[str, ...] = ("Packed", "Shipped", "Pending", "Delivered", "Returned")

GATES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # without I and O, which resemble digits
DIGITS = "0123456789"
