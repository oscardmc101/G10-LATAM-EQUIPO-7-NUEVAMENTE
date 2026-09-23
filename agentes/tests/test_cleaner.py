from agentes.rag.cleaner import clean_text


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_unicode_nfc():
    # Carácter compuesto 'e' con acento agudo
    decomposed = "e\u0301"
    composed = "\u00e9"
    result = clean_text(decomposed)
    assert result == composed


def test_clean_text_crlf_normalization():
    text = "linea1\r\nlinea2\rlinea3\nlinea4"
    result = clean_text(text)
    assert result == "linea1\nlinea2\nlinea3\nlinea4"


def test_clean_text_preserves_python_indentation():
    python_code = (
        "def calcular(x):\n"
        "    if x > 0:\n"
        "        total = x * 2\n"
        "        return total\n"
        "    return 0"
    )
    result = clean_text(python_code)
    assert result == python_code


def test_clean_text_preserves_yaml_indentation():
    yaml_content = (
        "server:\n"
        "  port: 8080\n"
        "  routes:\n"
        "    - path: /api/v1\n"
        "      enabled: true"
    )
    result = clean_text(yaml_content)
    assert result == yaml_content


def test_clean_text_preserves_markdown_trailing_spaces_and_tabs():
    # En Markdown, dos espacios al final de una línea indican un salto de línea (<br>)
    markdown_text = (
        "# Título  \n"
        "Párrafo con salto forzado.  \n"
        "\t- Elemento con tab"
    )
    result = clean_text(markdown_text)
    # Debe preservar exactamente los dos espacios finales de línea y el tabulador
    assert result == markdown_text


def test_clean_text_removes_invalid_control_characters():
    text_with_bad_chars = "Texto\x00con\x07caracteres\x1bde\x7fcontrol\tvalido\nlinea2"
    result = clean_text(text_with_bad_chars)
    assert result == "Textoconcaracteresdecontrol\tvalido\nlinea2"


def test_clean_text_collapses_excessive_newlines():
    text = "Sección 1\n\n\n\n\nSección 2"
    result = clean_text(text)
    assert result == "Sección 1\n\nSección 2"
