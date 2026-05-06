import os
import requests
from tqdm import tqdm
from urllib.parse import urlparse

TMP_DIR = "./__pycache__"
EN_US = os.getenv("LANG") != "zh_CN.UTF-8"


if EN_US:
    import huggingface_hub

    MODEL_DIR = huggingface_hub.snapshot_download(
        "Genius-Society/MiVOLO",
        cache_dir="./mivolo/__pycache__",
    )

else:
    import modelscope

    MODEL_DIR = modelscope.snapshot_download(
        "Genius-Society/MiVOLO",
        cache_dir="./mivolo/__pycache__",
    )


ZH2EN = {
    "性别年龄检测器": "Gender Age Detector",
    "上传模式": "Uploading Mode",
    "上传照片": "Upload Photo",
    "检测结果": "Detection Result",
    "存在儿童": "Has Child",
    "状态栏": "Status",
    "存在女性": "Has Female",
    "存在男性": "Has Male",
    "在线模式": "Online Mode",
    "网络图片链接": "Online Picture URL",
    "是": "Yes",
    "否": "No",
}


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


def is_url(s: str):
    try:
        # 解析字符串
        result = urlparse(s)
        # 检查scheme（如http, https）和netloc（域名）
        return all([result.scheme, result.netloc])

    except:
        # 如果解析过程中发生异常，则返回False
        return False


def download_file(url: str, save_path: str):
    if os.path.exists(save_path):
        print("目标已存在，无需下载")
        return

    create_dir(os.path.dirname(save_path))
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    # 使用 tqdm 创建一个进度条
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            progress_bar.update(len(data))

    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        os.remove(save_path)
        print("下载失败，重试中...")
        download_file(url, save_path)

    else:
        print("下载完成")

    return save_path


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_jpg_files(folder_path: str):
    all_files = os.listdir(folder_path)
    return [
        os.path.join(folder_path, file)
        for file in all_files
        if file.lower().endswith(".jpg")
    ]
