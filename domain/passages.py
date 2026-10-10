def chunk_text(text: str, max_chars: int = 1200) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start:
                end = newline + 1
        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            chunks.append({"start_offset": start + left, "end_offset": start + right,
                           "text": raw[left:right]})
        start = end
    return chunks
