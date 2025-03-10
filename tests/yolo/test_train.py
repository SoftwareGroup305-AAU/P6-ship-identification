import unittest
import os

from yolo.train import load_label_yolov8, train

class TestTrain(unittest.TestCase):
    def test_load_labels_yolov8(self):
        # Arrange
        expected_result = [[7, 0.69140625, 0.775, 0.5609375, 0.19609375], 
                         [10, 0.3609375, 0.3140625, 0.66328125, 0.28046875]] #based on test_label.txt
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_label_path = os.path.join(current_dir, "test_label.txt")

        #Act
        bounding_boxes = load_label_yolov8(test_label_path)

        #Assert
        self.assertEquals(expected_result, bounding_boxes)

if __name__ == "__main__":
    unittest.main()