import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
file_path = "E:/semester-1-2024-2025/HeThongThongMinh/TrafficSignDetection/Training/results-exp2.csv"  # Thay bằng đường dẫn file CSV của bạn
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()  # Làm sạch tên cột

# Các chỉ số cần vẽ
epochs = data["epoch"]
precision = data["metrics/precision"]
recall = data["metrics/recall"]
map_05 = data["metrics/mAP_0.5"]
map_05_095 = data["metrics/mAP_0.5:0.95"]

# Tính F1 score
f1_score = 2 * (precision * recall) / (precision + recall)

# Tạo đồ thị
plt.figure(figsize=(12, 6))

# Vẽ các đường đồ thị
plt.plot(epochs, precision, label="Precision", color="blue")
plt.plot(epochs, recall, label="Recall", color="green")
# plt.plot(epochs, f1_score, label="F1 Score", color="purple", linestyle="-", marker="o")
plt.plot(epochs, map_05, label="mAP@0.5", color="red")
plt.plot(epochs, map_05_095, label="mAP@0.5:0.95", color="orange")

# Thêm tiêu đề và nhãn
plt.title("Training Metrics over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.legend()
plt.grid(True)

# Hiển thị đồ thị
plt.show()
