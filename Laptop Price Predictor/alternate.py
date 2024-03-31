import streamlit as st
import pickle
import numpy as np

def indian_format(n):
    x = str(n).split('.')
    x1 = x[0]
    x2 = x[1] if len(x) > 1 else ''
    if len(x1) <= 3:
        return n if x2 == '' else f'{n}.{x2}'
    last_three = x1[-3:]
    rest = x1[:-3]
    rest = ','.join(reversed([rest[max(i-2,0):i] for i in range(len(rest), 0, -2)]))
    return f"{rest},{last_three}" if rest else last_three

pipe = pickle.load(open('model/pipe_object.pkl', 'rb'))
df = pickle.load(open('model/laptop_data.pkl', 'rb'))

st.title('Laptop Price Prediction')

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Basic Laptop Info")
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM(in GB)', [4, 6, 8, 12, 16, 24, 32, 64], index=2)
    weight = st.number_input('Weight of the Laptop(in Kg)', min_value=1.0, max_value=2.8, value=1.8, step=0.1)

with right_column:
    st.subheader("Advanced Laptop Info")
    
    left_advanced, right_advanced = st.columns(2)
    
    with left_advanced:
        touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])
        ips = st.selectbox('IPS', ['Yes', 'No'])
        screen_size = st.selectbox('Screen Size (in Inch)', [13.3, 15.6, 17.3])
        resolution = st.selectbox('Screen Resolution',
        [
    '1366 x 768',
    '1600 x 900',
    '1920 x 1080',
    '2304 x 1440',
    '2560 x 1440',
    '2560 x 1600',
    '2880 x 1800',
    '3200 x 1800',
    '3840 x 2160'
])
        os = st.selectbox('OS', df['os'].unique())

    with right_advanced:
        cpu = st.selectbox('CPU', df['Cpu Brand'].unique())
        gpu = st.selectbox('GPU', df['GpuBrand'].unique())
        hdd = st.selectbox('HDD(in GB)', [0, 256, 512, 1024, 2048])
        ssd = st.selectbox('SSD(in GB)', [0, 128, 256, 512, 1024])

if st.button('Predict Price'):
    touchscreen = int(touchscreen == 'Yes')
    ips = int(ips == 'Yes')
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    formatted_price = indian_format(predicted_price)
    st.title(f"\nPrice: ₹{formatted_price}")