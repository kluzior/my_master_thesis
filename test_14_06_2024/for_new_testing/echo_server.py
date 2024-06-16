import socket

def start_echo_server(host, port):
    # Tworzymy gniazdo serwera
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Ustawiamy opcję SO_REUSEADDR, aby ponownie użyć adresu po zamknięciu serwera
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bindowanie serwera do adresu IP i portu
    server_socket.bind((host, port))
    
    # Serwer zaczyna nasłuchiwać na maksymalnie 5 połączeń w kolejce
    server_socket.listen(5)
    
    print(f"Echo server nasłuchuje na {host}:{port}")
    
    while True:
        try:
            # Akceptujemy połączenie od klienta
            client_socket, client_address = server_socket.accept()
            print(f"Połączono z {client_address}")
            
            while True:
                try:
                    # Odbieramy dane od klienta
                    data = client_socket.recv(1024)
                    
                    if not data:
                        # Jeśli brak danych, zamykamy połączenie
                        break
                    
                    print(f"Otrzymano od klienta: {data}")
                    
                    # Odesłanie otrzymanych danych z powrotem do klienta
                    client_socket.sendall(data)
                
                except ConnectionResetError:
                    print("Połączenie zostało przerwane przez zdalnego hosta.")
                    break
            
            # Zamykamy połączenie z klientem
            client_socket.close()
            print(f"Rozłączono z {client_address}")

        except Exception as e:
            print(f"Wystąpił błąd: {e}")
            server_socket.close()
            break

if __name__ == "__main__":
    # Konfiguracja adresu IP i portu
    HOST = "127.0.0.1"  # Adres IP
    PORT = 30002        # Port

    # Uruchomienie serwera
    start_echo_server(HOST, PORT)
