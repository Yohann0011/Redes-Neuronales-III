# Redes Neuronales III

**Actividad:** Redes Neuronales III · Optimización, regularización y evaluación de modelos de deep learning
**Materia:** 604013/DEEP LEARNING -CONCEPTOS/FU/CAD2202023205/EIAIIPA2026
**Universidad de Cundinamarca** · Especialización en Inteligencia Artificial
**Docente:** Nidia Stella Garcia Roa
**Estudiante:** Yohann Javier Nikolay Salazar Cardenas · 2026
**Repositorio:** https://github.com/Yohann0011/Redes-Neuronales-III

Repositorio con las cinco prácticas desarrolladas en Google Colab sobre MNIST: partición
Train/Dev/Test, comparación de optimizadores, control del overfitting con L2 + Dropout + Early
Stopping, evaluación con métricas por clase y matriz de confusión, y réplica del entrenamiento con
PyTorch. **Sin redes convolucionales.**

---

## Contenido

| Notebook | Tema | Contenido principal |
|---|---|---|
| `01_Datos_Baseline_Keras.ipynb` | Datos, preprocesamiento y baseline | Partición estratificada Train/Dev/Test, normalización, distribución de clases, baseline 784-128-10 |
| `02_Optimizacion_Hiperparametros.ipynb` | Optimización e hiperparámetros | SGD + momentum, RMSprop y Adam en igualdad de condiciones; exploración del learning rate |
| `03_Regularizacion.ipynb` | Regularización y overfitting | Red grande sin regularización vs L2(1e-4) + Dropout(0.30) + Early Stopping (patience 2) |
| `04_Metricas_Evaluacion.ipynb` | Evaluación final sobre Test | Accuracy, Precision, Recall y F1 macro, reporte por clase, matriz de confusión y análisis de errores |
| `05_PyTorch_Opcional.ipynb` | Réplica con PyTorch | Ciclo explícito `zero_grad` → forward → loss → `backward` → `step`, Dropout y `weight_decay` |

Documentación y evidencias:

- `Informe_Redes_Neuronales_III.pdf` — documento técnico con código fuente, salidas de ejecución y análisis.
- `Informe_Redes_Neuronales_III_APA7.pdf` — versión del informe en formato APA 7.ª edición (portada, tabla de contenido, citaciones, referencias y apéndices con el código).
- `figuras/` — las 7 figuras generadas por los notebooks (curvas, learning rate, regularización, confusión y errores).
- `evidencias/` — salidas literales de consola de cada notebook y resumen de la ejecución.

---

## Resultados obtenidos

Valores reales de la ejecución documentada (semilla 42; decisiones tomadas sobre Dev; Test usado una sola vez).

| Experimento | Configuración | Métrica clave |
|---|---|---|
| Baseline | 784-128-10, Adam lr 0.001, 5 épocas | Dev loss 0.1058 · Dev acc **0.9685** |
| SGD + momentum | lr 0.01, momentum 0.9, 5 épocas | Dev acc 0.9627 |
| RMSprop | lr 0.001, 5 épocas | Dev acc **0.9723** (mejor de la comparación) |
| Adam | lr 0.001, 5 épocas | Dev acc 0.9698 |
| Learning rate (Adam, 3 épocas) | 1e-4 / 1e-3 / 1e-2 | Dev acc 0.9213 / **0.9675** / 0.9595 |
| Sin regularización | 535 818 parámetros, 15 épocas | Train loss 0.0002 vs Dev 0.1488 (overfitting visible) |
| L2 + Dropout + Early Stopping | parada en época 9 de 20 | Dev loss **0.1287** · Dev acc **0.9641** |
| Evaluación final en Test | modelo regularizado | Acc **0.9644** · P 0.9643 · R 0.9640 · F1 0.9641 |
| Confusión dominante | matriz 10×10 | 4→9 (20 casos); clase más débil: dígito 8 (F1 0.9489) |
| PyTorch | 784-256-10, 3 épocas | Dev acc 0.9690 · Test acc **0.9704** |

---

## Entorno de ejecución

| Componente | Versión |
|---|---|
| Python | 3.11 |
| TensorFlow / Keras | 2.21.0 / 3.15.1 (backend TensorFlow) |
| PyTorch | 2.14.1 |
| scikit-learn | 1.9.1 |
| Dataset | MNIST (60 000 / 10 000 imágenes de 28 × 28) |
| Particiones | Train 48 000 · Dev 12 000 · Test 10 000 (estratificadas, semilla 42) |

Los notebooks están pensados para **Google Colaboratory** (`Entorno de ejecución → Ejecutar todo`).

### Nota técnica sobre memoria

En entornos con ~1 GB de RAM la importación de TensorFlow (~620 MB) deja poco margen, por lo que los
entrenamientos se escriben como ciclos explícitos por lotes con `tf.GradientTape` normalizando solo el
lote en curso, y las evaluaciones se hacen por bloques. El procedimiento es matemáticamente idéntico a
`model.fit()`; la versión de alto nivel equivalente está documentada en el Notebook 01. El Notebook 05
carga `mnist.npz` con NumPy para no importar TensorFlow y PyTorch en el mismo kernel.

---

## Reproducir

```bash
# Abrir cualquier notebook en Colab (Archivo -> Subir notebook) y usar "Ejecutar todo".
# Semillas fijadas en 42 para reproducibilidad.
pip install numpy pandas matplotlib scikit-learn tensorflow-cpu torch
jupyter notebook
```

---

## Referencias

- Gulli, A., Kapoor, A., Pal, S., & Ciraci, S. (2023). *Deep Learning with TensorFlow and Keras* (3rd ed.). Packt Publishing.
- Stevens, E., Antiga, L., & Viehmann, T. (2020). *Deep Learning with PyTorch*. Manning Publications.
- Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR, 15*(1), 1929-1958.
- LeCun, Y., Cortes, C., & Burges, C. J. C. (2010). *MNIST handwritten digit database*. http://yann.lecun.com/exdb/mnist/
- Google (2026). *Google Colaboratory*. https://colab.research.google.com/
