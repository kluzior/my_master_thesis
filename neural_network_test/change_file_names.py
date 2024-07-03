import os

def rename_images(directory, x):
    # Pobierz listę plików w podanym katalogu
    files = os.listdir(directory)
    
    for file in files:
        # Sprawdź czy plik jest w formacie .png
        if file.endswith(".png"):
            # Rozdziel nazwę pliku na część numeru i rozszerzenie
            file_number, file_extension = os.path.splitext(file)
            
            try:
                # Przekształć część numeru na liczbę całkowitą
                number = int(file_number)
                
                # Zwiększ numer o x
                new_number = number + x
                
                # Utwórz nową nazwę pliku
                new_name = f"{new_number}{file_extension}"
                
                # Stwórz pełne ścieżki do starego i nowego pliku
                old_path = os.path.join(directory, file)
                new_path = os.path.join(directory, new_name)
                
                # Zmień nazwę pliku
                os.rename(old_path, new_path)
                print(f"Renamed {file} to {new_name}")
                
            except ValueError:
                # Jeśli nazwa pliku nie jest liczbą, pomiń plik
                print(f"Skipped {file}")

# Przykład użycia
directory = "neural_network_test/new/temp/"
x = 87  # Liczba, którą chcesz dodać do nazwy pliku
rename_images(directory, x)
