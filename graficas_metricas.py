import pandas as pd
import matplotlib.pyplot as plt
import os


csv_path = 'metricas_parcial3.csv'

# Leer datos
df = pd.read_csv(csv_path)

# Graficar detecciones YOLO
plt.figure(figsize=(10, 4))
plt.plot(df['YOLO_Detecciones'], label='Detecciones YOLO', color='green')
plt.xlabel('Frame')
plt.ylabel('Número de detecciones')
plt.title('Detecciones YOLO por frame')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'grafica_yolo_{os.path.splitext(csv_path)[0]}.png')
plt.show()

# Graficar coincidencias ORB
plt.figure(figsize=(10, 4))
plt.plot(df['ORB_Coincidencias'], label='Coincidencias ORB', color='blue')
plt.xlabel('Frame')
plt.ylabel('Número de coincidencias')
plt.title('Coincidencias ORB por frame')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'grafica_orb_{os.path.splitext(csv_path)[0]}.png')
plt.show()

# Comparación de ambas métricas
plt.figure(figsize=(10, 4))
plt.plot(df['YOLO_Detecciones'], label='YOLO', color='green')
plt.plot(df['ORB_Coincidencias'], label='ORB', color='blue')
plt.xlabel('Frame')
plt.ylabel('Cantidad')
plt.title('Comparación YOLO vs ORB')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'grafica_comparativa_{os.path.splitext(csv_path)[0]}.png')
plt.show()
