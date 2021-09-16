# Road-damage-segmentation

<h2>Введение</h2>

В данном проекте решена задача сегментация повреждений дорожного покрытия. Для этого использовались свёрточные нейросети. 
Также реализован алгоритм опредления размера повреждения.


<h2>Набор данных</h2>
 

Для обучения и тестирования разрабатываемого алгоритма собран
набор данных, представляющий собой размеченные вручную кадры с записи с видеорегистратора поездки по дорогам
Новосибирска, форматом 3840 на 2160 ppx. В наборе представлены различные дорожные ситуации: движение по трассе или городской дороге, виды
ограждений, обочин и разметки на проезжей части, наличие или отсутствие проезжающего транспорта разных габаритов, погодные условия, степень
разрушения дорожного покрытия. Набор изображений разделенна две части. В первой части разметка производилась на два класса, дорога и не
дорога. Во второй — повреждение и фон.

<p>
  <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/l_img11241.png" width="315" height="177" title="road">
  <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/l_road_mask_img11241.png" width="315" height="177" title="road_mask">
  <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/l_defect_mask_img11241.png" width="315" height="177" title="efect_mask">
</p>

Для улучшения качества работы моделей реализована аугментация. Исходное изображение зеркально отражалось повертикали и к нему с вероятностью 0.8 применялись цветовые преобразования:гамма-коррекция, изменение значений по цветовым каналам RGB (изменение происходит с вероятностью 0.5), а также яркость, контрастность
и насыщенность.

<h3>Картинки</h3>

<h2>Инструменты</h2>

<ul>
  <li>Data_Preprocessor</li>
  Класс Preprocessor Используется для преобразования иходных изображений в подходящий для нейросети вид. Изображения обезаются, уменьшаются и сохраняются в формате .png и .npy(преобразование в np.array и сохранение на этом этапе позволяет избежать этого на последующих этапах, что ускоряет обучение сетей). Также моно преобразовать видео. 
  Также включён модуль Augmentator для аугментации данных.
  <li>Data_Generator</li>
  Используется для подачи данных нейросети небольшим пакетами для того, чтобы не загружать ОП.
  <li>Callbacks_lib</li>
  Используется для вычисления и сохранения значения метрик(IoU,Precision, Recall, Dice index, Pixel Accuracy)
  в конце каждой эпохи на обучающей и валидационной выборках.
  <li>Road_models</li>
  Класс Unet_model для создания и обучения модели UNET для сегментации дорожного покрытия.
  Класс DeeplabV3_mv2_model vдля создания и обучения модели Deeplab v3 с энкодером mobilenet v2 для сегментации дорожного покрытия.
  <li>Defect_models</li>
  Класс для создания и обучения модели UNET для сегментации повреждений дорожного покрытия.
  <li>Evaluate_and_Inference</li>
  Класс визуализации работы модели. Позволяет получить предсказания,
  получить метрики качества для любой из подзадач - сегменации дорожного покрытия или повреждений на нём,
  при чём может быть использованно любое число моделей с любой степенью важности каждой. Метрики рассчитыватся
  как для отдельного изображения, так и средние значения для всего набора,
  строятся графики изменения метрик на оучающей и валидационной выборке.
  Также можно создать видео, отображающее предсказания модели.
  Для отдельного изображения можно получить карту активации.
  <li>Defect_area</li>
  Класс опредления площади повреждения дорожного покрытия.
 <li>Filtres</li>
 Некоторые фильтры для преобразования изображений.
</ul>


<h2>Результаты работы модели</h2>

<h3>Сегментация дорожного покрытия</h3>

<p>Для сегментации дороги была обучена нейросеть с архитектурой Deeplab v3 + с кодировщиком MobileNet v2, в качестве метода оптимизации использовался Adam.</p>
<p>На графиках прдставлено изменение функции потерь (верхний график) и индекса Дайса (нижний график) на обучающем и валидационном наборах в зависимости от эпохи.</p>

<p>
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/loss.png" height="260" width="800" title="dice">
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/dice.png" height="260" width="800" title="dice">
</p>

<p> 
 
</p>
Также была обучена нейросетьс архитектурой Unet, в качестве метода оптимизации использовался Adam.
<p>На графиках прдставлено изменение функции потерь (верхний график) и индекса Дайса (нижний график) на обучающем и валидационном наборах в зависимости от эпохи.</p>

<p>
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/loss_unet_road.png" height="260" width="800" title="dice">
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/dice_unet_road.png" height="260" width="800" title="dice">
</p>

<p> 
 
</p>
В ходе настройки обеих сетей, были опробованы изменения: числа слоёв сети, количества свёрточных фильтров на них, методов оптимизации и скорости обучения, функций потерь, нормализации по мини-батчу, регуляризации прореживания, количества эпох и т.д.

<p> 
 
 Сравнительная таблица для двух архитектур нейросетей, обученных с разными функциями потерь:
</p>

<table border="1" width="100%" cellpadding="5" bgcolor="black" cols = 5>
   <tr>
    <th rowspan="2">Функция потерь/метрика</th>
    <th colspan="2">Pixel Accuracy</th>
    <th colspan="2">Jaccard index</th>
   </tr>
   <tr>
    <td>Unet</td>
    <td>Deeplab v3</td>
    <td>Unet</td>
    <td>Deeplab v3</td>
  </tr>
 <tr>
    <td>Jaccard loss</td>
    <td>98.187</td>
    <td>98.489</td>
    <td>96.066</td>
    <td>97.993</td>
  </tr>
 <tr>
    <td>Jaccard loss 2</td>
    <td>98.194</td>
    <td>98.993</td>
    <td>96.112</td>
    <td>98.085</td>
  </tr>
 <tr>
    <td>Binary cross-entropy</td>
    <td>98.458</td>
    <td>98.937</td>
    <td>96.689</td>
    <td>98.006</td>
  </tr>
 </table>
<p>
 
 
Использование ансамбля взятием среднего по предсказаниям сетей
показывает лучшие результаты.
 <p>

</p>
 Значения метрик
 </p>

<table border="1" width="100%" cellpadding="5" bgcolor="black" cols = 2>
   <tr>
    <th>Метрика</th>
    <th>Значение</th>
   </tr>
   <tr>
    <td>IoU</td>
    <td>98.6</td>
  </tr>
 <tr>
    <td>Dice index</td>
    <td>99.3</td>
  </tr>
 <tr>
    <td>Взвешенная pixel accuracy</td>
    <td>99.2</td>
  </tr>
 <tr>
    <td>Pixel accuracy</td>
    <td>99.3</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>99.4</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>99.1</td>
  </tr>
 </table>
 
 Пример изображения,сегментации дороги на нём с помощью нейросети с архитектурой deeplab v3 + mobilenet v2 и с помощью нейросети с архитектурой unet, а также сегментация с использованием ансамблирования взвешенным этих двух архитектур (сверху вниз)
 
 <p>
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/img15553.png" height="240" width="700" title="dice">
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/deeplab_img15553.png" height="240" width="700" title="dice">
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/unet_img15553.png" height="240" width="700" title="dice">
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/combo_img15553.png" height="240" width="700" title="dice">
</p>
 
 <h3>Сегментация повреждений дорожного покрытия</h3>
Для сегментации повреждений дорожного покрытия была использована архитектура U-NET, с применением dropout регуляризации и нормализацией по
мини-батчам.
<p>На графиках прдставлено изменение функции потерь (верхний график) и индекса Дайса (нижний график) на обучающем и валидационном наборах в зависимости от эпохи.</p>

<p>
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/loss_unet_defect.png" height="260" width="800" title="dice">
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/dice_unet_defect.png" height="260" width="800" title="dice">
</p>
<p>

</p>
 Значения метрик
<table border="1" width="100%" cellpadding="5" bgcolor="black" cols = 2>
   <tr>
    <th>Метрика</th>
    <th>Значение</th>
   </tr>
   <tr>
    <td>IoU</td>
    <td>48.3</td>
  </tr>
 <tr>
    <td>Dice index</td>
    <td>72.3</td>
  </tr>
 <tr>
    <td>Взвешенная pixel accuracy</td>
    <td>91.1</td>
  </tr>
 <tr>
    <td>Pixel accuracy</td>
    <td>99.3</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>80.5</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>68.0</td>
  </tr>
 </table>
 
Пример сегментации повреждений дорожного покрытия с
использованием архитектуры U-Net на изображении из валидационного набора
данных.

<p>
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/img11241.png.png" height="240" width="700" title="dice">
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/defect_img11241.png" height="240" width="700" title="dice">
</p>

<h2>Определения площади повреждения</h2>

С помощью алгоритма было вычислено значение площали повреждений на изображении: 201.55 см
 <img src="https://github.com/Alinasas/Road-damage-segmentation/blob/master/readme_images/defect_img11241.png" height="240" width="700" title="dice">

