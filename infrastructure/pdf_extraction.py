from io import BytesIO

import pdfplumber
from pdfplumber.utils import cluster_objects, extract_text

from domain.structured_extraction import BoundingBox, PDFPage, TableRegion, TextBlock


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
