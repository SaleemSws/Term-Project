# Air Quality Prediction Dashboard | แดชบอร์ดทำนายคุณภาพอากาศ

## English

### Overview

This dashboard application provides predictive analytics for air quality metrics (PM10 and PM2.5) using machine learning models. The interactive dashboard is built with Dash and Plotly, featuring a sleek, dark pollution-themed design with dynamic visualizations.

### Features

- **7-Day PM2.5 Prediction**: Forecast PM2.5 levels for the next week with realistic daily fluctuations based on:
  - Weekday patterns (lower values on weekends)
  - Weather cycle simulations
  - Natural random variations
  - Initial environmental conditions

- **PM10 24-Hour Prediction**: Predict PM10 levels over the next 24 hours with diurnal patterns showing:
  - Morning and evening peaks
  - Afternoon and night dips
  - Based on current PM10 values and environmental factors

- **Interactive Visualizations**: Responsive graphs with informative annotations and tooltips

- **Modern UI**: Dark pollution-themed interface with:
  - Animated elements
  - Custom styling
  - Responsive design
  - Font Awesome integration

### Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```

### Required Files

The application requires two pre-trained machine learning models:
- `pm25_model_7d.pkl`: For 7-day PM2.5 predictions
- `catboost_pm10_24hV2.pkl`: For 24-hour PM10 predictions

### Running the Dashboard

```
python app.py
```

The application will be accessible at http://127.0.0.1:8050/

## Thai | ภาษาไทย

### ภาพรวม

แอปพลิเคชันแดชบอร์ดนี้ให้การวิเคราะห์เชิงทำนายสำหรับค่าคุณภาพอากาศ (PM10 และ PM2.5) โดยใช้โมเดลการเรียนรู้ของเครื่อง แดชบอร์ดแบบโต้ตอบสร้างด้วย Dash และ Plotly มาพร้อมกับการออกแบบธีมมลพิษแบบทันสมัยพร้อมการแสดงผลแบบไดนามิก

### คุณสมบัติ

- **การทำนาย PM2.5 แบบ 7 วัน**: คาดการณ์ระดับ PM2.5 สำหรับสัปดาห์ถัดไปด้วยความผันผวนรายวันที่สมจริง โดยอิงจาก:
  - รูปแบบวันในสัปดาห์ (ค่าต่ำลงในวันหยุดสุดสัปดาห์)
  - การจำลองวัฏจักรสภาพอากาศ
  - ความแปรปรวนแบบสุ่มตามธรรมชาติ
  - สภาวะแวดล้อมเริ่มต้น

- **การทำนาย PM10 แบบ 24 ชั่วโมง**: ทำนายระดับ PM10 ในช่วง 24 ชั่วโมงถัดไปด้วยรูปแบบรายวันที่แสดง:
  - ช่วงสูงสุดในตอนเช้าและตอนเย็น
  - ช่วงต่ำสุดในตอนบ่ายและกลางคืน
  - อิงจากค่า PM10 ปัจจุบันและปัจจัยสภาพแวดล้อม

- **การแสดงผลแบบโต้ตอบ**: กราฟที่ตอบสนองพร้อมคำอธิบายและเคล็ดลับที่มีข้อมูล

- **UI ทันสมัย**: อินเทอร์เฟซธีมมลพิษแบบมืดพร้อม:
  - องค์ประกอบเคลื่อนไหว
  - การจัดสไตล์ที่กำหนดเอง
  - การออกแบบที่ตอบสนอง
  - การรวม Font Awesome

### การติดตั้ง

1. โคลนที่เก็บนี้
2. ติดตั้งแพ็คเกจที่จำเป็น:
```
pip install -r requirements.txt
```

### ไฟล์ที่จำเป็น

แอปพลิเคชันต้องใช้โมเดลการเรียนรู้ของเครื่องที่ผ่านการฝึกฝนมาแล้วสองแบบ:
- `pm25_model_7d.pkl`: สำหรับการทำนาย PM2.5 แบบ 7 วัน
- `catboost_pm10_24hV2.pkl`: สำหรับการทำนาย PM10 แบบ 24 ชั่วโมง

### การเรียกใช้แดชบอร์ด

```
python app.py
```

สามารถเข้าถึงแอปพลิเคชันได้ที่ http://127.0.0.1:8050/ 