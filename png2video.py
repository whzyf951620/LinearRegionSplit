import cv2
import os

def main():
    data_path = "./patterns/"
    fps = 5  # 视频帧率
    size = (640, 480)  # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    for i in range(25):
        image_path = data_path + "%d.png" % i
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()