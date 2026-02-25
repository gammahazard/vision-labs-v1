"""Direct byte replacement for remaining 2 garbled chars"""
import os

p = r'c:\Users\mongo\Desktop\vision-labsv1\services\dashboard\static\generate.js'
with open(p, 'rb') as f:
    data = f.read()

# Line 625: garbled left arrow - hex from dump was: c3 a2 e2 80 a0
# Correct left arrow U+2190 = e2 86 90
# But wait, the arrow was "â†" which in the view showed as garbled.
# Let me find the exact bytes surrounding "Newer" to do a targeted replacement

# Find b'Newer</b' context
idx625 = data.find(b'Newer</b')
if idx625 == -1:
    # Try without closing tag
    idx625 = data.find(b'Newer</')
if idx625 >= 0:
    # Look backwards for the garbled bytes
    snippet = data[idx625-10:idx625+10]
    print(f'Around "Newer": {snippet.hex(" ")}')
    # Replace the garbled arrow + space before "Newer"
    # The garbled bytes before "Newer" should be the left arrow + space

# More targeted: find the exact garbled sequences by line content
lines = data.split(b'\r\n')

# Line 625 - find and show the exact non-ASCII bytes
line = lines[624]
print(f'L625 hex dump of non-ASCII:')
in_garble = False
start = 0
for j, b in enumerate(line):
    if b > 127 and not in_garble:
        in_garble = True
        start = j
    elif b <= 127 and in_garble:
        in_garble = False
        garbled = line[start:j]
        print(f'  bytes at {start}-{j}: {garbled.hex(" ")}')
        # Direct replacement
        if b' Newer' in line[j:j+10]:
            # This is the left arrow, replace with correct bytes
            correct = '\u2190'.encode('utf-8')  # ← = e2 86 90
            lines[624] = line[:start] + correct + line[j:]
            print(f'  -> replaced with left arrow')

# Line 865 - warning emoji
line = lines[864]
print(f'L865 hex dump of non-ASCII:')
in_garble = False
for j, b in enumerate(line):
    if b > 127 and not in_garble:
        in_garble = True
        start = j
    elif b <= 127 and in_garble:
        in_garble = False
        garbled = line[start:j]
        print(f'  bytes at {start}-{j}: {garbled.hex(" ")}')
        if b' Sweep' in line[j:j+10]:
            correct = '\u26a0\ufe0f'.encode('utf-8')  # ⚠️
            lines[864] = line[:start] + correct + line[j:]
            print(f'  -> replaced with warning sign')
        break

result = b'\r\n'.join(lines)
with open(p, 'wb') as f:
    f.write(result)

print(f'File size: {os.path.getsize(p)}')

# Verify
with open(p, 'r', encoding='utf-8') as f:
    vlines = f.readlines()
print(f'L625: {vlines[624].strip()[:80]}')
print(f'L865: {vlines[864].strip()[:80]}')
