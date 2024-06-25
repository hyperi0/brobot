import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

class ObjectDetector():
    def __init__(self,
                 model_name="facebook/detr-resnet-50"):
        self.processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")

    def detect_objects(self, img, threshold=0.9):
        inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        return results

    def start_capture(self, frameWidth=640, frameHeight=480):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, frameWidth)
        self.cap.set(4, frameHeight)
        self.cap.set(10,150)

    def stream_images(self):
        while self.cap.isOpened():
            success, img = self.cap.read()
                
            if success:
                cv2.imshow("Result", img)

                results = self.detect_objects(img)
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    print(
                            f"Detected {self.model.config.id2label[label.item()]} with confidence "
                            f"{round(score.item(), 3)} at location {box}"
                    )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break
                
if __name__ == "__main__":
    detector = ObjectDetector()
    detector.start_capture()
    detector.stream_images()