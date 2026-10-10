from io import BytesIO

import pdfplumber
from pdfplumber.utils import cluster_objects, extract_text

from domain.structured_extraction import BoundingBox, PDFPage, PDFToken, TableRegion, TextBlock


def extract_pdf(data: bytes) -> tuple[PDFPage, ...]:
    pages = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            grouped_chars = cluster_objects(page.chars, "doctop", 3)
            blocks = tuple(
                TextBlock(
                    id=f"p{number}-b{index}",
                    text=extract_text(chars).strip(),
                    bounds=BoundingBox(
                        min(char["x0"] for char in chars),
                        min(char["top"] for char in chars),
                        max(char["x1"] for char in chars),
                        max(char["bottom"] for char in chars),
                    ),
                )
                for index, chars in enumerate(grouped_chars, start=1)
                if extract_text(chars).strip()
            )
            tables = tuple(
                TableRegion(
                    id=f"p{number}-t{index}",
                    bounds=BoundingBox(*table.bbox),
                    rows=tuple(tuple(cell for cell in row) for row in table.extract()),
                )
                for index, table in enumerate(page.find_tables(), start=1)
            )
            pages.append(PDFPage(
                number=number,
                width=page.width,
                height=page.height,
                text=page.extract_text() or "",
                blocks=blocks,
                tables=tables,
            ))
    return tuple(pages)


def extract_pdf_tokens(data: bytes) -> tuple[tuple[PDFToken, ...], ...]:
    """Return exact word/character boxes, retaining the vertical issuer text."""
    result = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tokens = []
            for word in page.extract_words(return_chars=True):
                chars = word.get("chars", ())
                if not chars or not all(char.get("upright", True) for char in chars):
                    continue
                token_id = f"p{page_number}-w{len(tokens) + 1}"
                tokens.append(PDFToken(
                    token_id, word["text"],
                    BoundingBox(min(char["x0"] for char in chars), min(char["top"] for char in chars),
                                max(char["x1"] for char in chars), max(char["bottom"] for char in chars)),
                    tuple(f"{token_id}-c{index}" for index in range(len(chars))),
                    tuple(BoundingBox(char["x0"], char["top"], char["x1"], char["bottom"])
                          for char in chars),
                ))

            rotated = [char for char in page.chars if not char.get("upright", True)]
            for group in cluster_objects(rotated, "x0", 3):
                chars = sorted(group, key=lambda char: char["top"], reverse=True)
                if not chars:
                    continue
                token_id = f"p{page_number}-v{len(tokens) + 1}"
                tokens.append(PDFToken(
                    token_id, "".join(char["text"] for char in chars),
                    BoundingBox(min(char["x0"] for char in chars), min(char["top"] for char in chars),
                                max(char["x1"] for char in chars), max(char["bottom"] for char in chars)),
                    tuple(f"{token_id}-c{index}" for index in range(len(chars))),
                    tuple(BoundingBox(char["x0"], char["top"], char["x1"], char["bottom"])
                          for char in chars),
                ))
            result.append(tuple(tokens))
    return tuple(result)
