#!/usr/bin/python3

import pandas as pd

def compile_from_xsid(path):
    data = []
    xsid_id = None
    english = None
    dialect = None
    intent = None
    tokens = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith('# id:'):
                xsid_id = line[5:].strip()
            elif line.startswith('# text-en:'):
                english = line[10:].strip()
            elif line.startswith('# text:'):
                dialect = line[7:].strip()
            elif line.startswith('# intent:'):
                intent = line[9:].strip()
            elif line and not line.startswith('#'):
                pass
            elif line == "":  # end of a sample
                tokens.append((xsid_id, english, dialect, intent))
                if tokens:
                    data.extend(tokens)
                    tokens = []

    return pd.DataFrame(data, columns=["id", "text-en", "text", "intent"])


if __name__ == '__main__':
    df = compile_from_xsid('../data/de-by.test.conll')
    # print(df.head())
    # print(df.tail())

