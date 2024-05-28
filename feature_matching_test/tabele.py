import pandas as pd
from openpyxl import Workbook

# Lista nazw obrazów
images = [f'img{i}' for i in range(1, 8)]

# Lista nazw referencji i kategorii
references = [f'ref{i}' for i in range(1, 6)]
categories = ['raw', 'standard', 'otsu', 'adaptive']

# Stworzenie listy wierszy do DataFrame
rows = []
for ref in references:
    for category in categories:
        rows.append(f'{ref}_{category}')

# Stworzenie DataFrame o rozmiarze 7x20
data = pd.DataFrame(index=rows, columns=images)

# Utworzenie pliku Excel i zapisanie DataFrame
with pd.ExcelWriter('table.xlsx', engine='openpyxl') as writer:
    data.to_excel(writer, index=True, sheet_name='Sheet1')

print("Tabela została zapisana w pliku 'table.xlsx'")