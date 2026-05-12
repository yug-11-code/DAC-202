import re, glob, os

code_dir = r'C:\Users\Arman Srivastava\Desktop\Pillai Project\Code'
py_files = glob.glob(os.path.join(code_dir, '*.py'))

for fpath in py_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_docstring = False
    ds_char = None

    for line in lines:
        stripped = line.strip()

        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                ds_char = stripped[:3]
                if stripped.count(ds_char) >= 2 and len(stripped) > 3:
                    in_docstring = False
                else:
                    in_docstring = True
                new_lines.append(line)
                continue
        else:
            if ds_char in stripped:
                in_docstring = False
            new_lines.append(line)
            continue

        if stripped.startswith('#'):
            continue

        if '#' in line:
            in_str = False
            str_char = None
            result = []
            i = 0
            while i < len(line):
                c = line[i]
                if in_str:
                    result.append(c)
                    if c == str_char and (i == 0 or line[i-1] != '\\'):
                        in_str = False
                elif c in ('"', "'"):
                    in_str = True
                    str_char = c
                    result.append(c)
                elif c == '#':
                    break
                else:
                    result.append(c)
                i += 1
            line = ''.join(result).rstrip() + '\n'

        new_lines.append(line)

    final = []
    blank_count = 0
    for line in new_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                final.append(line)
        else:
            blank_count = 0
            final.append(line)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(final)

    orig = len(lines)
    new = len(final)
    print(f'{os.path.basename(fpath):25s}: {orig:4d} -> {new:4d} lines ({orig-new:+4d})')

print('\nDone! All comments removed.')
