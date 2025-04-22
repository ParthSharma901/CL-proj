import re

def clean_whatsapp_chat(chat_text):
    lines = chat_text.strip().split('\n')
    cleaned_lines = []
    pattern = r'^\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M - \+91 \d{5} \d{5}: (.*)$'

    for line in lines:
        if '<Media omitted>' in line or '<This message was edited>' in line or 'This message was deleted' in line:
            continue
        match = re.match(pattern, line)
        if match:
            cleaned_lines.append(match.group(1))
        else:
            if cleaned_lines:
                cleaned_lines[-1] += ' ' + line.strip()

    return '\n'.join(cleaned_lines)

with open("Whatsapp.txt", "r", encoding="utf-8") as f:
    chat_data = f.read()

cleaned_chat = clean_whatsapp_chat(chat_data)

with open("cleaned_chat.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_chat)

print("Cleaned chat saved to cleaned_chat.txt")
