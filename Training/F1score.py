import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
file_path = "E:/semester-1-2024-2025/HeThongThongMinh/TrafficSignDetection/Training/results.csv"  # Thay bằng đường dẫn file CSV của bạn
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()  # Làm sạch tên cột

# Các chỉ số cần thiết
epochs = data["epoch"]
precision = data["metrics/precision"]
recall = data["metrics/recall"]

# Tính F1 score
f1_score = 2 * (precision * recall) / (precision + recall)

# Tạo đồ thị
plt.figure(figsize=(12, 6))

# Vẽ đường F1 score
plt.plot(epochs, f1_score, label="F1 Score", color="purple", linestyle="-", marker="o")

# Thêm tiêu đề và nhãn
plt.title("F1 Score over Epochs", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Hiển thị đồ thị
plt.show()
