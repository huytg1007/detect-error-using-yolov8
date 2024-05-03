# Filename: find_dominant_color.py

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from datetime import datetime, date

pTime_1 = 0

# Luồng xử lý trong class này
# 1. Gọi Class khởi tạo các parameter cần thiết
# 2. Gọi hàm show_result()
# 2.1 Gọi hàm calculate_dominant_colors() để tìm màu chủ đạo của ảnh
# 2.2 Gọi hàm remove_black_clusters_at_corners() để xóa các pixel đen tại góc
# 2.3 Gọi hàm resize_resolution_image() để giảm độ phân giải để cải thiện hiệu năng
# 2.1 Gọi hàm find_longest_black_line() để tìm ra điểm dài nhất trong khu vực màu đen trong ảnh


class CalculateNGArea:
    def __init__(
        self,
        image,
        threshold_value,
        number_clusters=1,
        camera_name="",
        res_value=10,
        count=0,
    ):
        self.image = image
        self.threshold_value = threshold_value
        self.number_clusters = number_clusters
        self.camera_name = camera_name
        self.res_value = res_value
        self.count = count
        if self.image is None:
            raise ValueError("Image not found")

    # Tìm màu chủ đạo trong bức ảnh
    def calculate_dominant_colors(self):
        height, width, _ = np.shape(self.image)

        #  Chuyển đổi hình ảnh thành một mảng hai chiều
        data = np.reshape(self.image, (height * width, 3))

        # xài hàm cv2.kmeans thì phải chuyển đổi kiểu dữ liệu của mảng thành float32
        data = np.float32(data)

        # Định nghĩa tiêu chí dừng cho thuật toán k-means, kết hợp giữa số lượng lần lặp tối đa (10) và độ chính xác mong muốn (1.0).
        # Khi một trong hai tiêu chí được thỏa mãn, thuật toán sẽ dừng.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Áp dụng k-means clustering lên dữ liệu điểm ảnh, trả về nhãn cho mỗi điểm ảnh (labels) và tọa độ tâm của mỗi cluster (centers).
        # self.number_clusters chỉ định số lượng màu sắc chiếm ưu thế muốn tìm.
        _, labels, centers = cv2.kmeans(data, self.number_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # type: ignore

        # Đếm số lượng điểm ảnh trong mỗi cluster.
        _, counts = np.unique(labels, return_counts=True)

        # Tính phần trăm số điểm ảnh thuộc về mỗi cluster so với tổng số điểm ảnh, chuyển đổi sang tỷ lệ phần trăm.
        percentages = (counts / sum(counts)) * 100
        return centers, percentages

    # Tim ra vùng màu đen lớn nhất và xóa hết tất cả các black pixel còn lại
    def keep_largest_black_area(self, image):
        # chắc rằng hình hiện tại là binary
        if len(image.shape) != 2 or image.dtype != np.uint8:
            raise ValueError("Input must be a binary image.")

        # lưu ý phải là cv2.THRESH_BINARY_INV thì mới được
        _, threshold_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)

        # Tìm tất cả các contours trong ảnh
        contours, _ = cv2.findContours(
            threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Nếu không có contour nào được tìm thấy, trả về ảnh gốc
        if not contours:
            return image

        # dự vào tọa độ tìm được để
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y : y + h, x : x + w]

        cv2.imshow("roi", roi)
        return roi

    def find_longest_black_line(self, thresholded_image):
        # Tạo mảng tọa độ của các black point trong hình ảnh đã được threashold
        # và np.column_stack được sử dụng để chuyển đổi các chỉ số này thành một mảng của các cặp tọa độ (x, y).
        black_points = np.column_stack(np.where(thresholded_image == 0))

        # nếu chỉ có 1 diểm tọa độ thì không return về 0
        if len(black_points) < 2:
            return (0, 0), (0, 0), 0

        # dựa vào các cặp tọa độ tính ra mảng các khoảng cách
        distances = pdist(black_points, metric="euclidean")

        # tìm khoảng cách lớn nhất từ mảng các khoảng cách đã tính.
        max_distance = np.max(distances)

        # Xác định cặp điểm có khoảng cách lớn nhất để vẽ line
        # np.argmax(squareform(distances)) được sử dụng để tìm vị trí của khoảng cách lớn nhất trong mảng
        # squareform chuyển đổi mảng khoảng cách từ dạng vector sang ma trận vuông,
        # và np.unravel_index chuyển đổi chỉ số này sang chỉ số dòng và cột trong ma trận, tương ứng với cặp điểm trong black_points.
        max_dist_indices = np.unravel_index(
            np.argmax(squareform(distances)), squareform(distances).shape
        )

        point1 = black_points[max_dist_indices[0]]
        point2 = black_points[max_dist_indices[1]]

        return point1, point2, max_distance

    def resize_resolution_image(self, image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def remove_black_clusters_at_corners(self, image):
        # Assuming image is a binary image with black points and a white background
        # Make a copy of the image to preserve the original
        processed_image = image.copy()

        replacement_color = 255  # Set clusters to white

        # Get image dimensions
        height, width = processed_image.shape

        # Define corner points (top-left, top-right, bottom-left, bottom-right)
        corner_points = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
        ]

        # Perform floodFill from each corner with a small tolerance for black
        for point in corner_points:
            cv2.floodFill(
                processed_image,
                None,  # type: ignore
                point,
                replacement_color,  # type: ignore
                loDiff=(5, 5, 5, 5),
                upDiff=(5, 5, 5, 5),
                flags=cv2.FLOODFILL_FIXED_RANGE,
            )  # type: ignore

        return processed_image

    def show_result(self):
        global pTime_1
        pTime_1 = time.time()

        # Giảm độ phân giải hình ảnh để việc tìm điểm dài nhất dễ dàng hơn
        self.image = self.resize_resolution_image(self.image, self.res_value)

        # Tìm ra màu chủ đạo trong bức hình
        colors, _ = self.calculate_dominant_colors()

        list_dominant_colors = []

        for color in colors:
            list_dominant_colors.append(int(color[0]))

            # Phân ngưỡng hình ảnh dựa trên giá trị màu chủ đạo trừ đi threshhold ()
            _, thresholded_image = cv2.threshold(
                self.image,
                list_dominant_colors[0] - self.threshold_value,
                255,
                cv2.THRESH_BINARY,
            )

        # Chuyển dịnh dạng hình ảnh sang COLOR_BGR2GRAY
        result_image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)

        # Xóa toàn bộ các cụm pixel màu đen dính với các góc cạnh của ảnh
        result_image = self.remove_black_clusters_at_corners(result_image)

        # Tim ra vùng màu đen lớn nhất và xóa hết tất cả các black pixel còn lại
        # result_image = self.keep_largest_black_area(result_image)

        # Tìm điểm dài nhất của 2 pixel màu đen trên tấm hình phân ngưỡng
        point1, point2, max_distance = self.find_longest_black_line(result_image)

        if point1 is not None and point2 is not None:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

            cv2.line(
                result_image,
                (point1[1], point1[0]),
                (point2[1], point2[0]),
                (0, 255, 0),
                1,
            )

        result_image = cv2.resize(result_image, (200, 200))

        cv2.putText(
            result_image,
            f"{int(max_distance)}",
            (0, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
            1,
        )

        cv2.imshow(f"{self.camera_name} NG {self.count}", result_image)

        # curr_time_1 = time.time()
        # print(curr_time_1 - pTime_1)
        # fps = 1 / (curr_time_1 - pTime_1)
        # print(fps)

        # UNCOMMENT THIS CODE WHEN USE TEST IN THIS CLASS
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return point1, point2, max_distance


# TEST
# Example usage in the same file
if __name__ == "__main__":

    image = cv2.imread("d:/Downloads/Screenshot 2024-03-28 102140.png")
    finder = CalculateNGArea(image, 10, 1, "Test", 30, 1)
    # while True:
    result = finder.show_result()

    # Example usage
    # Use the function to keep the largest black area

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold_image = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y : y + h, x : x + w]
        area = cv2.contourArea(contour)
        print(area)
        cv2.rectangle(gray_image, (x, y), (x + w, y + h), (0, 0, 21), 2)
        cv2.putText(
            gray_image,
            "NG",
            (x + 5, y),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 0),
            1,
        )

    # Display the cropped image
    cv2.imshow("Cropped Image", roi)

    cv2.imshow("Bright Image", gray_image)
    cv2.imshow("Threshold Image", threshold_image)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask to fill in the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)

    # Fill in the rest of the black pixels with white
    result = cv2.bitwise_or(image, cv2.bitwise_not(mask))

    # Display the result
    cv2.imshow("Result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
