import re


def sanitize_filename(filename):
    """Sanitize filename by replacing special characters."""
    sanitized_filename = re.sub(r'[éè]', 'e', filename)
    sanitized_filename = re.sub(r'[à]', 'a', filename)
    sanitized_filename = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    return sanitized_filename