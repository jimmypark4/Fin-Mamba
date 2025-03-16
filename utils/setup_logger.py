import logging
import os

def setup_logger(log_dir):
    """
    Python logging 기본 설정:
    - 콘솔과 파일에 동시에 같은 로그 남김
    - 로그 파일: {log_dir}/training.log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 혹시 다른 Handler가 등록돼있으면 초기화

    # 1) 콘솔 핸들러 (기본)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)

    # 2) 파일 핸들러
    file_path = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)

    # 등록
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logger set up. Writing logs to {file_path}")
