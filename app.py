import os
import cv2
import imghdr
import shutil
import warnings
import numpy as np
import gradio as gr
from dataclasses import dataclass
from mivolo.predictor import Predictor
from utils import is_url, download_file, get_jpg_files, _L, MODEL_DIR, TMP_DIR


@dataclass
class Cfg:
    detector_weights: str
    checkpoint: str
    device: str = "cpu"
    with_persons: bool = True
    disable_faces: bool = False
    draw: bool = True


class ValidImgDetector:
    predictor = None

    def __init__(self):
        detector_path = f"{MODEL_DIR}/yolov8x_person_face.pt"
        age_gender_path = f"{MODEL_DIR}/model_imdb_cross_person_4.22_99.46.pth.tar"
        predictor_cfg = Cfg(detector_path, age_gender_path)
        self.predictor = Predictor(predictor_cfg)

    def _detect(
        self,
        image: np.ndarray,
        score_threshold: float,
        iou_threshold: float,
        mode: str,
        predictor: Predictor,
    ) -> np.ndarray:
        predictor.detector.detector_kwargs["conf"] = score_threshold
        predictor.detector.detector_kwargs["iou"] = iou_threshold
        if mode == "Use persons and faces":
            use_persons = True
            disable_faces = False

        elif mode == "Use persons only":
            use_persons = True
            disable_faces = True

        elif mode == "Use faces only":
            use_persons = False
            disable_faces = False

        predictor.age_gender_model.meta.use_persons = use_persons
        predictor.age_gender_model.meta.disable_faces = disable_faces
        detected_objects, out_im = predictor.recognize(image)
        has_child, has_female, has_male = False, False, False
        if len(detected_objects.ages) > 0:
            has_child = _L("是") if min(detected_objects.ages) < 18 else _L("否")
            has_female = _L("是") if "female" in detected_objects.genders else _L("否")
            has_male = _L("是") if "male" in detected_objects.genders else _L("否")

        return out_im[:, :, ::-1], has_child, has_female, has_male

    def valid_img(self, img_path):
        image = cv2.imread(img_path)
        return self._detect(image, 0.4, 0.7, "Use persons and faces", self.predictor)


def infer(photo: str):
    status = "Success"
    result = child = female = male = None
    try:
        if is_url(photo):
            if os.path.exists(TMP_DIR):
                shutil.rmtree(TMP_DIR)

            photo = download_file(photo, f"{TMP_DIR}/download.jpg")

        detector = ValidImgDetector()
        if not photo or not os.path.exists(photo) or imghdr.what(photo) == None:
            raise ValueError("请正确输入图片")

        result, child, female, male = detector.valid_img(photo)

    except Exception as e:
        status = f"{e}"

    return status, result, child, female, male


warnings.filterwarnings("ignore")
gr.TabbedInterface(
    interface_list=[
        gr.Interface(
            fn=infer,
            inputs=gr.Image(label=_L("上传照片"), type="filepath"),
            outputs=[
                gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                gr.Image(
                    label=_L("检测结果"),
                    type="numpy",
                    buttons=["download", "fullscreen"],
                ),
                gr.Textbox(label=_L("存在儿童")),
                gr.Textbox(label=_L("存在女性")),
                gr.Textbox(label=_L("存在男性")),
            ],
            examples=get_jpg_files(f"{MODEL_DIR}/examples"),
            flagging_mode="never",
            cache_examples=False,
        ),
        gr.Interface(
            fn=infer,
            inputs=gr.Textbox(label=_L("网络图片链接"), buttons=["copy"]),
            outputs=[
                gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                gr.Image(
                    label=_L("检测结果"),
                    type="numpy",
                    buttons=["download", "fullscreen"],
                ),
                gr.Textbox(label=_L("存在儿童")),
                gr.Textbox(label=_L("存在女性")),
                gr.Textbox(label=_L("存在男性")),
            ],
            flagging_mode="never",
        ),
    ],
    tab_names=[_L("上传模式"), _L("在线模式")],
    title=_L("性别年龄检测器"),
).launch(css="#gradio-share-link-button-0 { display: none; }", ssr_mode=False)
