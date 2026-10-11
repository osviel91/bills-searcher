import time
from pathlib import Path

from application.extractor_registry import TEMPLATE_DETECTORS
from application.template_detection import detect_template
from infrastructure.pdf_extraction import extract_pdf


WIDTH, HEIGHT = 595, 842


def _pdf(template):
    if template == "T01":
        columns = (30, 120, 220, 390, 480, 575)
        rows = (300, 340, 420, 500, 635)
        contents = (
            (0, 310, "CONCEPTO"), (1, 310, "ALBARAN"), (2, 310, "FECHA"),
            (3, 310, "T.CHEQUES"), (4, 310, "TOTAL"),
            (0, 365, "SERVICIO SINTETICO"), (1, 365, "ALB-TEST-01"),
            (2, 365, "01/01/2026"), (4, 365, "12,00"),
            (0, 445, "COMISION"), (0, 475, "FORFAIT GESTION"),
            (0, 535, "GASTOS ENVIO TOTAL EUROS"),
        )
    elif template == "T02":
        columns = (30, 105, 180, 255, 330, 410, 490, 575)
        rows = (280, 320, 385, 450, 515, 572)
        header = TEMPLATE_DETECTORS[1].headers
        contents = tuple((index, 290, value) for index, value in enumerate(header)) + (
            (0, 350, "ARTICULO SINTETICO"), (1, 350, "1"), (2, 350, "10,00"),
            (3, 350, "10,00"), (4, 350, "21%"), (5, 350, "2,10"),
            (0, 475, "BASE IMPONIBLE"), (0, 500, "CUOTA IVA"),
            (0, 525, "TOTAL FACTURA TOTAL A PAGAR"),
        )
    else:
        columns, rows, contents = (), (), ((0, 100, "DOCUMENTO SINTETICO SIN ESTRUCTURA CONOCIDA"),)

    commands = ["0.5 w"]
    for x in columns:
        commands.extend((f"{x} {HEIGHT - rows[0]} m", f"{x} {HEIGHT - rows[-1]} l", "S"))
    for y in rows:
        commands.extend((f"{columns[0]} {HEIGHT - y} m", f"{columns[-1]} {HEIGHT - y} l", "S"))
    for column, top, text in contents:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        x = columns[column] + 3 if columns else 45
        commands.extend(("BT /F1 8 Tf", f"{x} {HEIGHT - top - 10} Td", f"({safe}) Tj", "ET"))
    stream = "\n".join(commands).encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] "
         "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>").encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    )
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, item in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + item + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    data.extend(b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:]))
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(data)


def fixture(template):
    data = _pdf(template)
    detected = detect_template(extract_pdf(data))
    if detected.template != template:
        raise AssertionError(f"synthetic {template} fixture detected as {detected.template}")
    return data


def child_failure(send, _hook_dir):
    try:
        raise RuntimeError
    except RuntimeError as error:
        send.send({"status": "worker_error", "error_type": type(error).__name__})


def child_timeout(_send, _hook_dir):
    time.sleep(30)


def child_stale(send, hook_dir):
    hook_dir = Path(hook_dir)
    (hook_dir / "shadow-child-started").touch()
    while not (hook_dir / "shadow-child-release").exists():
        time.sleep(0.02)
    send.send({"status": "succeeded", "template": "T01", "fields_produced": 2,
               "lines_produced": 1, "warnings": 0})
