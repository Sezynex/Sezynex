import cv2

def main():
    # USB kamera bağlantısını başlat
    cam = cv2.VideoCapture(0)

    while True:
        # Kameradan bir frame al
        ret, frame = cam.read()
        if not ret:
            print("Kamera bağlantısı başarısız.")
            break

        # Alınan frame'i ekranda göster
        cv2.imshow("USB Kamera", frame)

        # Çıkış için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kullanılan kaynakları serbest bırak
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
