import cv2
import numpy as np
from openvino.runtime import Core
import requests




def convert_result_to_image(rgb_image, resized_image, boxes, threshold=0.3, conf_labels=True):

    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    (real_y, real_x), (resized_y, resized_x) = rgb_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    for box in boxes:

        conf = box[-1]
        if conf > threshold:
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]


            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)


            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    # f"{conf:.2f}",
                    f"{box[-1]:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                 )

    return rgb_image



def get_file(link):
    file_name = link.split('/')[-1]
    r = requests.get(link, allow_redirects = True)
    open(file_name, "wb").write(r.content)


if __name__ == '__main__':

    # bin_link = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/person-detection-asl-0001/FP16/person-detection-asl-0001.bin"
    # xml_link = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/person-detection-asl-0001/FP16/person-detection-asl-0001.xml"
    # get_file(xml_link)
    # get_file(bin_link)


    ie = Core()
    ir_model = ie.read_model("person-detection-asl-0001.xml", "person-detection-asl-0001.bin")
    compiled_model = ie.compile_model(ir_model, device_name="CPU")
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output("boxes")
    N, C, H, W = input_layer_ir.shape


    cap = cv2.VideoCapture("A Day In The Park.mp4")


    while True:
        ret, frame = cap.read()

        resized_frame = cv2.resize(frame, (H, W))
        input_img = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
        boxes = compiled_model([input_img])[output_layer_ir]
        boxes = boxes[~np.all(boxes == 0, axis=1)]


        input_img = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

        boxes = compiled_model([input_img])[output_layer_ir]
        boxes = boxes[~np.all(boxes == 0, axis=1)]
        result_frame = convert_result_to_image(frame, resized_frame, boxes, conf_labels=False)

        cv2.imshow("Frame", result_frame)

        key = cv2.waitKey(30)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
