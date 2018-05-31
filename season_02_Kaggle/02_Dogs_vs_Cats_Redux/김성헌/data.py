import cv2
import numpy as np
import math

class Dataset_image:
    def __init__(self, dataset, batch_size):
        """데이터 관리

        Args:
            dataset: [image_files, labels]
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._batch_idx = 0
        self._dataset_count = len(dataset)
        self._data_count = len(dataset[0])
        self._image_shape = [150, 150, 3]
        self._resize = False

    def setImageInfo(self, image_shape, resize=False):
        """이미지 정보 설정

        Args:
            image_shape: [height, width, channel]
            resize: True, False. 이미지 축소수행 유무
        """
        self._image_shape = image_shape
        self._resize = resize

    def init_iterator(self):
        self._batch_idx = 0

    def countData(self):
        return self._data_count

    def countBatch(self):
        return math.ceil(self.countData() / self._batch_size)

    def getBatchRange (self, data_size, batch_size, batch_idx):
        begin_idx = batch_idx * batch_size
        end_idx = begin_idx + batch_size
        end_idx = min(end_idx, data_size)
        return begin_idx, end_idx

    def getImageData (self, image_file_list, image_shape=[150, 150, 3], resize=False):
        data = np.ndarray((len(image_file_list), image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8)
        for i, image_file in enumerate(image_file_list):
            if resize == False:
                data[i] = cv2.imread(image_file)
            else:
                data[i] = cv2.resize(cv2.imread(image_file), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        return data

    def next_batch(self):
        begin_idx, end_idx = self.getBatchRange(self._data_count, self._batch_size, self._batch_idx)
        self._batch_idx += 1

        batch_x = self._dataset[0][begin_idx : end_idx]
        batch_x_image = self.getImageData(batch_x, self._image_shape, self._resize)

        if self._dataset_count == 1:
            return batch_x_image
        elif self._dataset_count == 2:
            batch_y = self._dataset[1][begin_idx : end_idx]
            batch_y = np.reshape(batch_y, [-1,2])
            return batch_x_image, batch_y
