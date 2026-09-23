import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Limpieza conservadora de texto:
    - Normalización Unicode con NFC (para no descomponer caracteres técnicos).
    - Normalización segura de saltos de línea (CRLF y CR a LF).
    - Eliminación de caracteres de control inválidos/no imprimibles, preservando tabuladores y newlines.
    - Reducción de 3 o más saltos de línea consecutivos a 2.
    - PRESERVA indentación y espacios significativos (código Python, YAML, dobles espacios Markdown).
    - NO ejecuta strip() ni rstrip() línea por línea.
    """
    if not text:
        return ""

    # Normalización Unicode estándar (NFC preserva caracteres compuestos técnicos)
    text = unicodedata.normalize("NFC", text)

    # Normalización uniforme de saltos de línea
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Eliminación de caracteres de control ASCII inválidos excepto \t (0x09) y \n (0x0a)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Reducción de 3 o más saltos de línea consecutivos a 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Retirar saltos de línea vacíos al inicio y final del documento sin tocar espacios horizontales
    return text.strip("\n")