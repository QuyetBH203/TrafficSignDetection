# TRAFFIC SIGN DETECTION 

## Enviroment
Python version: >= 3.10

## Setup


### Step 1
```bash
git clone https://github.com/QuyetBH203/TrafficSignDetection.git
cd yolov5
```
### Step 2
Tạo môi trường ảo trong python và kích hoạt môi trưởng ảo
```bash
python -m venv myenv
myenv\Scripts\activate
```
### Step 3
Cài đặt các pakages cần thiết

```bash
pip install -r requirements.txt
```
### Step 4
copy ảnh vào thư mục yolov5\images, ví dự copy ảnh có tên là camdungxe.png, chạy lệnh sau để nhận diện

```bash
python detect.py --weights models\best.pt --source images\camdungxe.png --conf 0.5

```
 