import glob
import os
from pathlib import Path

import cv2
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_data_numbers_and_percentages():
    data_dir = "dataset/UCF-25"
    with open('dataset/classes.txt', 'r') as f:
        classes = f.read().splitlines()

    data_counts = []
    total_count = 0

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        file_count = len(os.listdir(class_dir))
        data_counts.append(file_count)
        total_count += file_count
    print(total_count)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=data_counts)
    plt.xticks(rotation=90)
    plt.xlabel("Sınıflar")
    plt.ylabel("Veri Sayısı")
    plt.title("Her Sınıf İçin Veri Sayısı")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.pie(data_counts, labels=classes, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
    plt.axis('equal')
    plt.title("Her Sınıf İçin Veri Yüzdelik Dağılımı")
    plt.show()


def calculate_frame_sampling_parameters():
    data_dir = "dataset/UCF-25"
    with open('dataset/classes.txt', 'r') as f:
        classes = f.read().splitlines()

    total_frames = 0
    total_fps = 0
    video_count = 0
    for cls in classes:
        cls_dir = Path(data_dir) / cls.strip()
        avi_files = glob.glob(f"{cls_dir}/*.avi")
        for file in avi_files:
            cap = cv2.VideoCapture(file)
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_fps += cap.get(cv2.CAP_PROP_FPS)
            video_count += 1

    average_video_duration = round(total_frames / total_fps, 2)
    average_fps = round(total_fps / video_count, 2)

    frames_to_sample = int(average_video_duration * 2)
    sampling_step = int(average_fps / 2)

    print("Ortalama video süresi: ", average_video_duration, "Ortalama FPS: ", average_fps)
    print("Örneklenecek kare sayısı: ", frames_to_sample, "Örnekleme adımı: ", sampling_step)


calculate_frame_sampling_parameters()
