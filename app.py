from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import logging
from datetime import datetime

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# โหลดโมเดลที่ train ไว้
model = None
model_path = os.path.join(os.path.dirname(__file__), "trained_model.sav")

def load_model():
    global model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("โมเดลโหลดสำเร็จ")
        return True
    except Exception as e:
        logger.error(f"ไม่สามารถโหลดโมเดลได้: {str(e)}")
        return False

# โหลดโมเดลเมื่อเริ่มต้นแอพ
load_model()

@app.route('/')
def index():
    return jsonify({
        "message": "Diabetes Prediction API พร้อมใช้งาน!",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "model_loaded": model is not None
    })

@app.route('/health')
def health_check():
    """Health check endpoint สำหรับ monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model is not None else "not_loaded"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ตรวจสอบว่าโมเดลโหลดแล้วหรือไม่
        if model is None:
            return jsonify({
                'error': 'โมเดลไม่ได้ถูกโหลด กรุณาลองใหม่อีกครั้ง',
                'status': 'model_not_loaded'
            }), 500

        # ตรวจสอบ Content-Type
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type ต้องเป็น application/json',
                'status': 'invalid_content_type'
            }), 400

        data = request.get_json()
        
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if not data:
            return jsonify({
                'error': 'ไม่พบข้อมูล JSON',
                'status': 'no_data'
            }), 400

        # ตรวจสอบ required features
        required_features = ['Age', 'BMI', 'BloodPressure', 'GlucoseLevel', 
                           'InsulinLevel', 'FamilyHistory', 'PhysicalActivity']
        
        missing_features = [feature for feature in required_features if feature not in data]
        if missing_features:
            return jsonify({
                'error': f'ข้อมูลไม่ครบถ้วน ขาด: {", ".join(missing_features)}',
                'required_features': required_features,
                'status': 'missing_features'
            }), 400

        # ตรวจสอบ data types และ ranges
        validation_errors = []
        
        # Age: 0-120
        if not isinstance(data.get('Age'), (int, float)) or not (0 <= data['Age'] <= 120):
            validation_errors.append('Age ต้องเป็นตัวเลขระหว่าง 0-120')
        
        # BMI: 10-60
        if not isinstance(data.get('BMI'), (int, float)) or not (10 <= data['BMI'] <= 60):
            validation_errors.append('BMI ต้องเป็นตัวเลขระหว่าง 10-60')
        
        # BloodPressure: 80-200
        if not isinstance(data.get('BloodPressure'), (int, float)) or not (80 <= data['BloodPressure'] <= 200):
            validation_errors.append('BloodPressure ต้องเป็นตัวเลขระหว่าง 80-200')
        
        # GlucoseLevel: 50-300
        if not isinstance(data.get('GlucoseLevel'), (int, float)) or not (50 <= data['GlucoseLevel'] <= 300):
            validation_errors.append('GlucoseLevel ต้องเป็นตัวเลขระหว่าง 50-300')
        
        # InsulinLevel: 0-500
        if not isinstance(data.get('InsulinLevel'), (int, float)) or not (0 <= data['InsulinLevel'] <= 500):
            validation_errors.append('InsulinLevel ต้องเป็นตัวเลขระหว่าง 0-500')
        
        # FamilyHistory: 0 or 1
        if data.get('FamilyHistory') not in [0, 1]:
            validation_errors.append('FamilyHistory ต้องเป็น 0 หรือ 1')
        
        # PhysicalActivity: 0 or 1
        if data.get('PhysicalActivity') not in [0, 1]:
            validation_errors.append('PhysicalActivity ต้องเป็น 0 หรือ 1')

        if validation_errors:
            return jsonify({
                'error': 'ข้อมูลไม่ถูกต้อง',
                'validation_errors': validation_errors,
                'status': 'validation_failed'
            }), 400

        # เตรียมข้อมูลสำหรับ prediction
        input_features = np.array([[
            data['Age'],
            data['BMI'],
            data['BloodPressure'],
            data['GlucoseLevel'],
            data['InsulinLevel'],
            data['FamilyHistory'],
            data['PhysicalActivity']
        ]])

        # ทำนาย
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0]
        
        # ส่งผลลัพธ์
        result = {
            'prediction': int(prediction),
            'probability': {
                'no_diabetes': float(probability[0]),
                'diabetes': float(probability[1])
            },
            'result_text': 'มีความเสี่ยงเป็นเบาหวาน' if prediction == 1 else 'ไม่มีความเสี่ยงเป็นเบาหวาน',
            'confidence': float(max(probability)),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Prediction successful: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'เกิดข้อผิดพลาดในการทำนาย',
            'details': str(e),
            'status': 'prediction_failed'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'ไม่พบ endpoint ที่ร้องขอ',
        'status': 'not_found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'HTTP method ไม่ได้รับอนุญาต',
        'status': 'method_not_allowed'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์',
        'status': 'internal_error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)