import cv2
import time
import darknet
import threading
import queue
import os


def draw_boxes(detections, image, source_shape):
    s_w, s_h = source_shape  # source, eg 416x416
    t_h, t_w, _ = image.shape  # target
    w_scale = float(t_w) / s_w
    h_scale = float(t_h) / s_h

    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        x = x * w_scale
        y = y * h_scale
        w = w * w_scale
        h = h * h_scale
        left, top, right, bottom = darknet.bbox2points((x, y, w, h))

        if float(confidence) > 0.6:
            # Definir valores para ampliar a região de corte
            ampliacao = 40  # Valor para ampliar as coordenadas

            # Ajustar as coordenadas para ampliar a região de corte
            top = max(0, top - ampliacao)
            bottom = min(image.shape[0], bottom + ampliacao)
            left = max(0, left - ampliacao)
            right = min(image.shape[1], right + ampliacao)

            # Realizar o corte e salvar a imagem
            cv2.imwrite("imagem1.jpg", image[
                top:bottom, left:right])

            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def receive(rtsp_cam, id_camera, frames_queue):
    cap = cv2.VideoCapture(rtsp_cam, cv2.CAP_FFMPEG)
    while True:
        ret, frame = cap.read()
        if ret:
            # frame = cv2.resize(frame, (1280, 720))
            try:
                frames_queue.put_nowait([frame, id_camera])
            except queue.Full:
                pass


if __name__ == '__main__':
    network, class_names, colors = darknet.load_network("yolo/yolov4-tiny-face.cfg", "yolo/coco.data",
                                                        "yolo/yolov4-tiny-face.weights")

    rtsp_urls = [
        {'url': 'rtsp://192.168.0.82:8554/', 'id': 0},
        # Adicione mais URLs RTSP conforme necessário
    ]
    caps = [cv2.VideoCapture(url['url'], cv2.CAP_FFMPEG) for url in rtsp_urls]

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    frames_queues = [queue.Queue(maxsize=10) for _ in rtsp_urls]

    # Inicie threads para receber os frames das streams RTSP
    threads = []
    for url in rtsp_urls:
        thread = threading.Thread(target=receive, args=(
            url['url'], url['id'], frames_queues[url['id']]))
        thread.start()
        threads.append(thread)

    windows = {}  # Dicionário para armazenar as janelas

    while True:
        t1 = time.time()  # Marque o tempo inicial

        for idx, frames_queue in enumerate(frames_queues):
            try:
                frame, id_camera = frames_queue.get(timeout=1)
                # recorta a parte central da imagem
                (x1, y1), (x2, y2) = [(314, 279), (1006, 638)]
                frame = frame[y1-10:y2, x1-20:x2+20]
                cap = caps[id_camera]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height),
                                           interpolation=cv2.INTER_LINEAR)
                darknet.copy_image_from_bytes(
                    darknet_image, frame_resized.tobytes())

                detections = darknet.detect_image(
                    network, class_names, darknet_image, thresh=0.6)
                frame = draw_boxes(detections, frame, (width, height))

                if id_camera not in windows:
                    # cv2.namedWindow(f'Detect {id_camera}', cv2.WINDOW_NORMAL)
                    windows[id_camera] = True

                cv2.imshow(f'Detect {id_camera}', frame)

            except queue.Empty:
                pass

        dt = time.time() - t1  # Tempo decorrido entre loops
        fps = 1 / dt  # Cálculo do FPS
        print(f'FPS: {fps:.2f}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            os._exit(0)
            break

    for cap in caps:
        cap.release()

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
