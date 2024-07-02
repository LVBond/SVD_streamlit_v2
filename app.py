import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO


def svd_compression(img, top_k):
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    compressed_img = (U[:, :top_k] @ np.diag(s[:top_k]) @ Vt[:top_k, :])
    return compressed_img


def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


st.title('Сингулярный преобразователь')
st.divider()

uploaded_image = st.sidebar.file_uploader('Закинь свою картинку')
button = st.sidebar.button('Мне повезёт!')

if 'image' not in st.session_state:
    st.session_state.image = None

if button:
    st.session_state.image = load_image('https://variety.com/wp-content/uploads/2021/07/Rick-Astley-Never-Gonna-Give-You-Up.png')

if uploaded_image:
    st.session_state.image = Image.open(uploaded_image)

if st.session_state.image is not None:
    image_raw = st.session_state.image
    image_array = np.array(image_raw)
    image_gray = np.array(image_raw.convert('L'))

    st.caption(f'Разрешение изображения: {image_array.shape[0]}x{image_array.shape[1]}')
    st.image(image_array, caption='Оригинал', use_column_width=True)

    with st.sidebar.form(key='compression_form'):
        top_k = st.slider('Выберите top K для сжатия', min_value=1, max_value=min(image_gray.shape))
        color_choice = st.toggle('Цветное')
        submit_button = st.form_submit_button(label="Сжать")

    if submit_button:
        st.caption(f'Вы выбрали top K: {top_k}')
        with st.spinner(text="Сжатие..."):
            if not color_choice:
                compressed_image = svd_compression(image_gray, top_k)
            else:
                channels = [image_array[:, :, i] for i in range(3)]
                compressed_channels = [svd_compression(channel, top_k) for channel in channels]
                compressed_image = np.stack(compressed_channels, axis=-1)

            compressed_image = (compressed_image - np.min(compressed_image)) / (np.max(compressed_image) - np.min(compressed_image))
            st.image(compressed_image, caption='Сжатое изображение', use_column_width=True)





# #  5. Разложите матрицу по SVD, как в нашем примере из лекции сегодня

# U, sing_values, V = np.linalg.svd(image) # метод возвращает три матрицы
# sigma = np.zeros(shape = image.shape)
# np.fill_diagonal(sigma, sing_values)
# # plt.imshow(U@sigma@V, cmap='grey')


# # 6. Выберите топ k сингулярных чисел и схлопните матрицу обратно

# top_k = k
# trunc_U = U[:, :top_k]
# trunc_sigma = sigma[:top_k, :top_k]
# trunc_V = V[:top_k, :]
# # trunc_U.shape, trunc_sigma.shape, trunc_V.shape


# # 7. Посмотрите результат
# # 8. Выберите минимальный k при котором картинка еще все так же различима
# # 9. Какую долю этот k составляет от всех сингулярных чисел
# # fig, ax = plt.subplots(1, 2, figsize=(15, 10))

# new_image = trunc_U@trunc_sigma@trunc_V
# # Если получилось получить сжатое изображение, показываем его
# if new_image is not None:
#     image = Image.open(new_image)
#     st.title("Uploaded Image")
#     st.image(image, caption='Uploaded Image', use_column_width=True)



# # 10. Обернуть в стримлит-приложение, с интерфейсом, в котором пользователь 
# # может подгрузить свою картинку и выбрать количество сингулярных чисел. 
# # В ответ получить сжатый вариант

